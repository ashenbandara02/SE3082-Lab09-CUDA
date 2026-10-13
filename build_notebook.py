"""Reproduce the lab's supplied cells and fill in the exercise solutions."""
from pathlib import Path
import re
import pymupdf
import nbformat as nbf

root = Path(__file__).parent
doc = pymupdf.open(root / 'Lab09_CUDA_Colab_SE3082.pdf')
lines = []
for page in doc:
    for block in page.get_text('dict')['blocks']:
        for line in block.get('lines', []):
            spans = [s for s in line['spans'] if s['font'].startswith('DejaVuSansMono')]
            if spans:
                lines.append(''.join(s['text'] for s in spans).rstrip())

programs = {}
current = None
for line in lines:
    if line.startswith('%%writefile '):
        current = line.split()[1]
        programs[current] = []
    elif line.startswith('!'):
        current = None
    elif current:
        programs[current].append(line)

programs.pop('name.cu', None)  # Workflow illustration, not an exercise.
for name in programs:
    programs[name] = '\n'.join(programs[name]).strip() + '\n'

s = programs['ex5_vecmul.cu']
s = s.replace('    int i = 0;', '    int i = blockIdx.x * blockDim.x + threadIdx.x;', 1)
s = s.replace('    c[i] = a[i] * b[i];', '    if (i < n)\n        c[i] = a[i] * b[i];')
s = s.replace('int blocks = 0;', 'int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;')
s = s.replace('// TODO 4: launch vecMul with <<<blocks, THREADS_PER_BLOCK>>>', 'vecMul<<<blocks, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, N);')
s = s.replace('// TODO 5: copy the result d_c back into c on the host', 'cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);')
s = s.replace('// TODO 6: print the last 1000 results as "index) a x b = c"', 'for (int i = N - 1000; i < N; i++)\n        printf("%d) %d x %d = %d\\n", i, a[i], b[i], c[i]);')
s = s.replace('// TODO 7: verify all N results on the CPU and print the number of mismatches', 'int errors = 0;\n    for (int i = 0; i < N; i++)\n        if (c[i] != a[i] * b[i]) errors++;\n    printf("Verification: %d mismatches out of %d\\n", errors, N);')
programs['ex5_vecmul.cu'] = s
s = programs['ex6_matrix.cu']
s = s.replace('int row = 0;', 'int row = blockIdx.y * blockDim.y + threadIdx.y;')
s = s.replace('int col = 0;', 'int col = blockIdx.x * blockDim.x + threadIdx.x;')
s = re.sub(r'    int idx = 0;[^\n]*\n    c\[idx\] = a\[idx\] \* b\[idx\];[^\n]*', '    if (row < rows && col < cols) {\n        int idx = row * cols + col;\n        c[idx] = a[idx] * b[idx];\n    }', s)
s = s.replace('dim3 numBlocks(1, 1);', 'dim3 numBlocks((COLS + threadsPerBlock.x - 1) / threadsPerBlock.x,\n                   (ROWS + threadsPerBlock.y - 1) / threadsPerBlock.y);')
s = s.replace('// TODO 5: launch mulElem with <<<numBlocks, threadsPerBlock>>>', 'mulElem<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, ROWS, COLS);')
s = s.replace('// TODO 6: copy d_c back into c on the host', 'cudaMemcpy(c, d_c, size, cudaMemcpyDeviceToHost);')
s = s.replace('// TODO 7: print C[9999][9000] ... C[9999][9999]  (the last 1000 elements)', 'for (int col = COLS - 1000; col < COLS; col++) {\n        size_t idx = (size_t)(ROWS - 1) * COLS + col;\n        printf("C[%d][%d]) %d x %d = %d\\n", ROWS - 1, col, a[idx], b[idx], c[idx]);\n    }')
s = s.replace('// TODO 8: verify all elements on the CPU and print the number of mismatches', 'size_t errors = 0;\n    for (size_t i = 0; i < count; i++)\n        if (c[i] != a[i] * b[i]) errors++;\n    printf("Verification: %zu mismatches out of %zu\\n", errors, count);')
programs['ex6_matrix.cu'] = s

for name, source in programs.items():
    source = re.sub(r'// TODO \d+:', '// Solution:', source)
    source = re.sub(r'\s*// <-- (replace|guard) this line', '', source)
    programs[name] = source
    (root / name).write_text(source, encoding='utf-8')

nb = nbf.v4.new_notebook()
cells = nb.cells
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))
def program(name):
    code('%%writefile ' + name + '\n' + programs[name])
    stem = Path(name).stem
    code(f'!nvcc -arch=sm_75 {name} -o {stem} && ./{stem}')

md('# SE3082 — Lab 09: Introduction to CUDA\n\n**Student ID:** IT24100315\n\nBased on Lab09_CUDA_Colab_SE3082.pdf (Semester 2, 2026).\n\n**Execution status:** Prepared but not yet executed on a Colab GPU. Expected outputs below are reference values from the lab sheet, not recorded execution results. Run all cells with a T4 GPU and save the resulting notebook before submission.')
md('## Exercise 1 — Set up CUDA in Google Colab\n\nOpen this notebook in Colab. Choose **Runtime → Change runtime type → T4 GPU → Save**. Run the diagnostic cells and record the GPU name, compute capability and compiler release from their output. The compile cells use `sm_75` for T4; for L4 use `sm_89`, A100 `sm_80`, or H100 `sm_90` in every compile cell. If `nvcc` is unavailable, use `/usr/local/cuda/bin/nvcc`.')
code('!nvidia-smi')
code('!nvcc --version')
code('!nvidia-smi --query-gpu=name,compute_cap --format=csv')
program('hello.cu')
md('Expected: four GPU greetings for thread IDs 0–3, in any order.\n\nColab files are temporary; reconnect and run all cells to rebuild them. The notebook itself can be saved to Drive.')
md('## Exercise 2 — Add two integers\n\nOne block and one thread add 2 and 7. Expected: `Result is 9`.')
program('ex2_add.cu')
md('### Think — answers\n\n1. In this program, ordinary host variables reside in CPU memory. `cudaMalloc` allocates separate GPU storage accessible to the kernel; explicit copies transfer the values.\n2. `&a` is the address of the host integer to copy; `d_a` already contains the address of the device allocation.\n3. `add<<<1, 1>>>` runs one thread. `add<<<1, 10>>>` runs ten threads, but they would all write to the same `c`, creating a data race even though they compute the same value. Use separate output elements or guard the scalar write to thread 0.')
md('## Exercise 3 — 512 blocks, one thread per block\n\nPredict: 512 blocks × 1 thread = 512 threads. Each block needs a warp with only one active lane, leaving 31 lanes unused. Expected final line: `Verification: 0 mismatches out of 512`.')
program('ex3_blocks.cu')
md('## Exercise 4 — One block, 512 threads\n\n`threadIdx.x` selects each element. Expected: `0) 83 + 77 = 160`, last result `511) 67 + 6 = 73`, and zero mismatches. A single block has 16 full warps, but cannot distribute work across multiple multiprocessors.')
program('ex4_threads.cu')
md('### Experiment — 2,048 threads\n\nThe following cells make a temporary copy with N = 2048, compile and run it, then remove it. The original Exercise 4 remains at N = 512. Expected launch error: `Kernel launch failed: invalid configuration argument`, reported by `cudaGetLastError()`, because 2048 exceeds the 1024 threads-per-block limit. Result lines after a failed launch are invalid because no results were computed.')
code("from pathlib import Path\np = Path('ex4_threads.cu')\nPath('ex4_limit.cu').write_text(p.read_text().replace('#define N 512', '#define N 2048'))")
code('!nvcc -arch=sm_75 ex4_limit.cu -o ex4_limit && ./ex4_limit')
code("from pathlib import Path\nfor name in ('ex4_limit.cu', 'ex4_limit'):\n    Path(name).unlink(missing_ok=True)")
md('## Exercise 5 — Multiply 10,000,000 vector elements\n\nCompleted global index, bounds guard, rounded-up block count, kernel launch, copy back, last 1,000 results and full CPU verification.\n\nExpected: 19,532 blocks × 512 = 10,000,384 threads; the final block has 128 active threads and 384 guarded threads. The printed results run from `9999000) 76 x 52 = 3952` to `9999999) 31 x 73 = 2263`, followed by `Verification: 0 mismatches out of 10000000`.')
program('ex5_vecmul.cu')
md('### Think — answers\n\nA multiple of 32 fills complete warps. With 500 threads per block, 20,000 blocks launch exactly 10,000,000 threads, but each block uses 16 warps with 12 unused lanes in its last warp. A bounds guard is still appropriate for arbitrary array sizes.')
md('## Exercise 6 — Element-wise 10,000 × 10,000 matrix product\n\nColumns use x; rows use y. Flat index is `row * cols + col`. Both coordinates are guarded. Blocks have 32 × 16 = 512 threads; the rounded-up grid is 313 × 625 = 195,625 blocks and 100,160,000 threads. The extra 160,000 threads are guarded. Each matrix occupies 400 MB; three matrices require 1.2 GB on both host and device.\n\nExpected first printed element: `C[9999][9000]) 29 x 56 = 1624`; last: `C[9999][9999]) 86 x 21 = 1806`; verification: zero mismatches out of 100,000,000.')
program('ex6_matrix.cu')
md('## Finishing checklist\n\n- Run all cells on a Colab T4 GPU and inspect the actual outputs.\n- Confirm hello prints four greetings, Exercise 2 gives 9, and Exercises 3–6 report zero mismatches.\n- Confirm the Exercise 4 limit experiment reports the expected launch failure.\n- Save the notebook with outputs visible; download the `.ipynb` and submit or share as instructed.\n- Choose **Runtime → Disconnect and delete runtime** to release the GPU.')
nb.metadata = {'colab': {'name': 'SE3082_Lab09_IT24100315.ipynb', 'provenance': []}, 'accelerator': 'GPU', 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}}
nbf.validate(nb)
nbf.write(nb, root / 'SE3082_Lab09_IT24100315.ipynb')
print(f'Generated {len(programs)} CUDA files and {len(cells)} notebook cells.')
