# How to Migrate a Python ML Project to Java

Lessons learned from porting OpenAI Whisper (Python) to whisper4j (pure Java 26).

---

## The Core Principle

**Layer-by-layer numerical equivalence is the only reliable way to port a neural network.**

You cannot port an entire pipeline and debug it end-to-end. The output will be wrong, and you won't know which of the
50+ operations introduced the error. Instead, you build a parallel validation harness in the source language that prints
intermediate values at every stage, then match those values in the destination language one stage at a time.

---

## Architecture

```
Source Project (Python)              Destination Project (Java)
─────────────────────────────────    ─────────────────────────────────
reference_repos/whisper/             src/main/java/...
    │                                    │
    ├── validate_against_reference.py    ├── ValidateStages.java
    │   (prints values at each stage)    │   (prints values at each stage)
    │                                    │
    │   Stage 1: mel spectrogram ──────► Stage 1: mel spectrogram
    │   Stage 2: conv1 + gelu ─────────► Stage 2: conv1 + gelu
    │   Stage 3: conv2 + gelu ─────────► Stage 3: conv2 + gelu
    │   Stage 4: encoder output ───────► Stage 4: encoder output
    │   Stage 5: decoder logits ───────► Stage 5: decoder logits
    │                                    │
    └── (ground truth values)            └── (must match within tolerance)
```

The Python script and the Java test class use the **same model weights** and the **same audio input**. They print the
first N values at each stage. You advance to the next stage only when the current stage matches.

---

## Phase 1: Study the Source Project's History

Before writing any code, read the source project's commit history end-to-end. The OpenAI Whisper repo has 167 commits
spanning 3 years. The bugs they fixed are the bugs you will re-introduce if you don't understand them.

### Critical lessons from the Whisper commit history:

**Correctness bugs that took months to find:**

- `9f70a35` — "Fix attention caching to make it actually work" — KV cache was silently broken from the initial commit.
  The model produced output, but cross-attention keys/values were being recomputed every step instead of cached. This
  means the initial implementation was *correct but slow*, and the "optimization" (caching) was broken for months.
- `520796a` — "fix token suppression" — Suppression logic was wrong, causing the model to generate tokens it shouldn't.
- `ec1b34b` — "fix compression ratio function" — The quality gate that triggers temperature fallback was computing the
  wrong ratio.
- `76148a5` — "suppress generating non-timestamp tokens at the beginning" — Without this, the model could start a
  segment with text instead of a timestamp, breaking the seek-based chunking.

**Hallucination and repetition (the hardest class of bugs):**

- `919a713` — "attempt to fix the repetition/hallucination issue" — Changed from zero-padding the spectrogram to
  zero-padding the audio before mel extraction. This single change in *where* padding happens fixed a major
  hallucination issue.
- `38f2f4d` — "fix all_tokens handling that caused more repetitions" — A subtle bug in how tokens were accumulated
  across segments caused the model to repeat itself.
- `ba3f3cd` — "Skip silence around hallucinations" — Added `hallucination_silence_threshold` to detect and skip
  hallucinated segments by checking for silence in the audio.
- `248b6cb` — "fix condition_on_previous_text" — Cross-segment context was being applied incorrectly, causing
  hallucination cascades.
- `90db0de` — "Bugfix: Illogical avoid computing higher temperatures on no_speech" — The no-speech detection was
  conflating compression-ratio failures with actual silence, skipping temperature fallback when it was needed.

**Architecture evolved, not designed:**

- `68e44bd` → `9323b25` → `5380767` — Three commits to get QK attention weight extraction right. First attempt stored it
  as a side effect on the module. Reverted. Then returned it as a second output. This pattern (try → revert → redesign)
  is normal.
- `b91c907` — "Avoid rearranging all caches" — Cross-attention KV cache was being recomputed every decoder step. Fixed
  to compute once and reuse. This is the same bug whisper4j had to fix independently.
- `27f9713` — "using sdpa if available" — PyTorch's `scaled_dot_product_attention` replaced the manual QK^T → softmax →
  @V pattern. The manual path was kept as fallback. Your port must implement the manual path (you don't have SDPA).

**Timestamp heuristics are fragile:**

- `f572f21` — "Improve timestamp heuristics" — Added `last_speech_timestamp` tracking to handle pauses correctly. This
  was commit #120 of 167 — timestamp logic was refined for over a year.
- `e8622f9` — "word timing tweaks" — DTW alignment needed multiple rounds of tuning after the initial implementation.
- `2b0c297` — "Don't update duration if last timestamp is same as begin" — Edge case in timestamp parsing that caused
  zero-length segments.

### What this means for your port:

1. **Read every bug-fix commit in the source repo before porting that component.** If they fixed KV caching 6 months
   after initial commit, you will get it wrong too unless you port the fixed version.
2. **The initial commit is not the correct implementation.** It's the starting point. The correct implementation is
   HEAD.
3. **Hallucination bugs are not in the neural network — they're in the pipeline logic.** Padding, token accumulation,
   temperature fallback, and cross-segment context are where hallucinations come from.
4. **Timestamp logic will take multiple iterations.** Don't try to get it perfect on the first pass.

---

## Phase 2: Establish Ground Truth

Before writing any destination code, build a standalone validation script in the source language that:

1. Loads the model weights directly (not through the framework's high-level API)
2. Implements each operation manually (conv, attention, FFT, etc.)
3. Prints intermediate tensor values at every stage boundary
4. Uses a fixed, deterministic input (same audio file, same prompt tokens)

This script is your oracle. Every value it prints becomes a test assertion.

**Why manual ops instead of the framework?** Because the framework hides details you need to match exactly — padding
conventions, bias broadcasting, activation function variants, weight layout (row-major vs column-major). When your port
doesn't match, you need to know which specific operation diverged.

### Example: `tools/validate_against_reference.py`

```python
# Stage 2: Conv1 + GELU
conv1_out = gelu(conv1d(mel_3d, W['encoder.conv1.weight'],
                        W['encoder.conv1.bias'], stride=1, padding=1))
print(f"Conv1+GELU first4={conv1_out.flat[:4]}")

# Stage 3: Conv2 + GELU
conv2_out = gelu(conv1d(conv1_out, W['encoder.conv2.weight'],
                        W['encoder.conv2.bias'], stride=2, padding=1))
print(f"Conv2+GELU first4={conv2_out.flat[:4]}")
```

Each `print` becomes a reference value hardcoded in the Java validation test.

---

## Phase 3: Port Bottom-Up, One Layer at a Time

Port in dependency order. Each component must pass numerical validation before you build on top of it.

### Recommended order for neural network projects:

1. **Tensor / linear algebra primitives** — matmul, add, reshape, transpose
2. **Signal processing** — FFT, mel spectrogram, resampling
3. **Weight loading** — parse the model file format, extract tensors
4. **Atomic neural ops** — Linear, LayerNorm, Conv1d, GELU, softmax
5. **Composite blocks** — MultiHeadAttention, ResidualAttentionBlock
6. **Encoder** — stack of blocks + pre/post processing
7. **Decoder** — token embedding, cross-attention, autoregressive loop
8. **Pipeline** — audio → mel → encoder → decoder → text

At each level, write a validation test that compares output against the Python reference values.

### Example: `ValidateStages.java`

```java
static final float[] REF_CONV1 = {0.03736455f, 0.00012353f, 0.02271019f, 0.05902079f};

static final float TOL_CONV = 0.01f;

// Stage 2: Conv1 + GELU
Conv1d conv1 = new Conv1d(
        weights.get("encoder.conv1.weight"),
        weights.get("encoder.conv1.bias"), 1, 1
);

Tensor c1 = conv1.forward(mel3d).gelu();

assertFirst4("Conv1+GELU",c1, REF_CONV1, TOL_CONV);
```

---

## Phase 4: Tolerance Budgets

Floating-point differences compound across layers. Define tolerance tiers:

| Stage depth               | Typical tolerance | Why                                |
|---------------------------|-------------------|------------------------------------|
| Single op (conv, linear)  | `< 0.001`         | FP32 rounding only                 |
| After encoder (6+ layers) | `< 0.05`          | Error compounds per layer          |
| Decoder logits            | `< 0.5`           | 12+ layers of compounding          |
| Top-K token IDs           | **exact match**   | The final output must be identical |

If a stage exceeds its tolerance budget, the bug is in that stage — not downstream. Fix it before proceeding.

---

## Phase 5: Common Bugs That Stage Validation Catches

These are the bugs that were found in whisper4j by comparing stage outputs. Every one of them would have been invisible
in an end-to-end test (the model would just produce garbage text).

| Bug                                    | Symptom at stage         | Root cause                                                                      |
|----------------------------------------|--------------------------|---------------------------------------------------------------------------------|
| Wrong GELU variant                     | Conv+GELU off by ~5%     | Used `sigmoid` approximation instead of `tanh` (must match training)            |
| Conv1d bias broadcasting               | Conv output wrong        | Bias cycled every 512 elements instead of broadcasting per-channel              |
| Mel filter source                      | Mel spectrogram off      | Computed mel filters from scratch instead of using model's pre-computed filters |
| Weight layout (col-major vs row-major) | Linear output transposed | GGML stores weights column-major; Java expects row-major                        |
| FFT precision                          | Mel spectrogram off      | Direct DFT for n=400 (non-power-of-2) had accumulated rounding error            |
| Positional embedding offset            | Decoder output wrong     | Off-by-one in positional embedding indexing during autoregressive decode        |

---

## Phase 6: Multi-Format Validation

Once the pipeline works with one model format, validate that other formats produce identical stage outputs. Use the
first format as ground truth.

```java
// ValidateFormats.java — same 5 stages, multiple model formats
Map<String, Path> models = new LinkedHashMap<>();
models.

put("GGML",Path.of("ggml-base.en.bin"));
        models.

put("SafeTensors",Path.of("model.safetensors"));
        models.

put("PyTorch",Path.of("pytorch_model.bin"));

        for(

var entry :models.

entrySet()){

validateFormat(entry.getKey(),entry.

getValue(),audio);
        }
```

This catches weight name mapping bugs, dtype conversion errors, and shape mismatches between formats.

---

## Phase 7: Pipeline Logic (Where Hallucinations Live)

The neural network is the easy part. The pipeline logic around it — temperature fallback, token suppression, seek-based
chunking, cross-segment context — is where the hard bugs live. The Whisper source project spent more commits fixing
pipeline logic than fixing the model itself.

### Port these in order, testing each independently:

1. **Basic greedy decode** — single 30s chunk, no fallback
2. **Temperature fallback** — retry with higher temperature on compression ratio > 2.4 or avg logprob < -1.0
3. **No-speech detection** — skip chunks where the model predicts silence
4. **Token suppression** — suppress SOT, task tokens, blank tokens at position 0
5. **Timestamp tokens** — enforce timestamp pairs, non-decreasing order
6. **Seek-based chunking** — advance seek position based on last timestamp token
7. **Cross-segment context** — condition_on_previous_text (disabled by default — causes hallucination cascades on small
   models)
8. **Word-level timestamps** — cross-attention DTW alignment (experimental in the source project too)

### Key insight from the source history:

The source project's hallucination fix (`919a713`) changed *where* audio padding happens — padding the raw audio before
mel extraction instead of padding the spectrogram after. This is a one-line change that took months to identify. Your
port must match the final padding behavior exactly.

---

## Phase 8: Performance (Only After Correctness)

Optimization is a separate phase. The commit history shows a clear pattern:

1. **Get it correct first** — even if it takes 55 seconds for 30s of audio
2. **Profile to find the actual bottleneck** — don't guess
3. **Optimize one thing at a time** — validate correctness after each change
4. **Document what didn't work** — prevent re-attempting failed approaches

### Optimization log pattern (from `OPTIMIZATION_NOTES.md`):

```
### [Name] ([result])
- Attempted: [what you tried]
- Result: [measured impact]
- Why: [root cause of success or failure]
- Lesson: [what to do or avoid next time]
```

This is critical when working with AI assistants — without this log, the assistant will repeatedly suggest the same
failed optimizations.

### Performance lessons from the source project:

The source project's own optimization history mirrors ours:

- `b91c907` — Avoid rearranging all KV caches (same optimization we needed)
- `27f9713` — Use SDPA when available (we can't — no PyTorch, so we implement the manual path)
- The source project never optimized the mel spectrogram — it uses numpy FFT. We had to optimize it (Bluestein FFT)
  because Java's pure-math FFT was 23x slower.

---

## Project Structure

```
project/
├── reference_repos/          # Source project(s), checked in as-is
│   └── whisper/              # Read-only reference for diffing
├── tools/
│   └── validate_against_reference.py  # Ground truth generator
├── src/main/java/            # Destination implementation
├── src/test/java/
│   ├── ValidateStages.java   # Stage-by-stage numerical comparison
│   └── ValidateFormats.java  # Cross-format equivalence
├── implementation_plan.md    # Task breakdown with status
└── OPTIMIZATION_NOTES.md     # What worked, what didn't, why
```

### Key files and their roles:

- **`reference_repos/`** — The source project, unmodified. You read it constantly during porting. Having it in-repo
  means your AI assistant can reference it directly.
- **`tools/validate_against_reference.py`** — The oracle. Runs the source implementation step-by-step and prints
  intermediate values. You run this once to generate reference values, then hardcode them in the Java tests.
- **`ValidateStages.java`** — The destination-side mirror. Same stages, same input, same print format. When a stage
  doesn't match, the bug is in that stage's Java implementation.
- **`implementation_plan.md`** — Living document. Mark tasks complete as you go. This prevents re-doing work and gives
  the AI assistant context about what's done.
- **`OPTIMIZATION_NOTES.md`** — Append-only log of every optimization attempted. Prevents circular optimization
  attempts.

---

## Rules

1. **Never skip a stage.** If stage 3 is wrong, stages 4–7 are meaningless.
2. **Same weights, same input, same output.** Use the exact same model file and audio file in both languages.
3. **Print, don't assert, during development.** Print first-N values so you can eyeball the magnitude of divergence.
   Convert to assertions once the stage passes.
4. **One bug per commit.** When you find a bug via stage validation, fix it and commit immediately. The commit message
   should name the stage and the root cause.
5. **Keep the reference repos in-tree.** Your AI assistant needs to read the source implementation to understand what
   each operation should do.
6. **Activation functions must match training exactly.** There is no "close enough" for GELU, softmax, or layer norm. A
   1% error in an activation compounds to garbage output after 12 layers.
7. **Weight layout is the first thing to verify.** Row-major vs column-major, channel-first vs channel-last, transposed
   vs not — get this right before debugging anything else.
8. **Optimize only after correctness.** A fast wrong answer is worse than a slow right answer. The whisper4j encoder
   went from 55s to 0.2s, but only after it produced correct text.
9. **Read the source project's bug-fix commits before porting each component.** The bugs they fixed are the bugs you
   will re-introduce.
10. **Pipeline logic causes hallucinations, not the neural network.** Padding, token accumulation, temperature fallback,
    and cross-segment context are where the hard bugs live.
