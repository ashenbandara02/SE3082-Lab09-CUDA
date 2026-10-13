# SE3082 Lab 09 — CUDA

Student ID: **IT24100315**.

The notebook follows `Lab09_CUDA_Colab_SE3082.pdf`: all six exercises, the supplied lecture programs, completed vector/matrix kernels, the 2048-thread experiment, and answers to the discussion questions.

## Run in Google Colab

1. Open `SE3082_Lab09_IT24100315.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Select **Runtime → Change runtime type → T4 GPU → Save**.
3. Select **Runtime → Run all**. The notebook writes, compiles and runs every CUDA program.
4. Check the actual output: four hello messages, `Result is 9`, and zero mismatches in Exercises 3–6. The separate Exercise 4 limit experiment should report an invalid launch configuration.
5. Save/download the notebook with outputs visible, then disconnect and delete the runtime.

Compile cells use `-arch=sm_75` as required by the sheet. If assigned another GPU, change every compile flag using the architecture table in Exercise 1.

## Files

- `SE3082_Lab09_IT24100315.ipynb`: submission notebook.
- `hello.cu`, `ex2_add.cu`, `ex3_blocks.cu`, `ex4_threads.cu`: supplied programs.
- `ex5_vecmul.cu`: 10,000,000-element product, 512 threads/block, 19,532 blocks, bounds guard, last 1,000 results, complete verification.
- `ex6_matrix.cu`: element-wise 10,000 × 10,000 product, 32 × 16 threads/block, 313 × 625 blocks, both bounds guarded, last 1,000 elements, complete verification.
- `build_notebook.py`: regenerate notebook and source files from the supplied PDF; requires `pymupdf` and `nbformat`.

## Execution status

The notebook has not been executed on a Colab GPU. Reference outputs are clearly labelled as expected values; no GPU execution results have been fabricated. Local validation checks notebook structure, source-cell consistency and CPU execution of the extracted kernel logic, including out-of-range threads. CUDA compilation and GPU execution remain to be performed in Colab before submission.
