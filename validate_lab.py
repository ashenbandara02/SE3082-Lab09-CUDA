"""Check notebook consistency and run actual kernel bodies on CPU edge cases.

This is a logic check, not a CUDA compiler or GPU execution substitute.
"""
from pathlib import Path
import re
import subprocess
import nbformat

root = Path(__file__).parent
nb = nbformat.read(root / 'SE3082_Lab09_IT24100315.ipynb', as_version=4)
nbformat.validate(nb)
written = 0
for cell in nb.cells:
    if cell.cell_type == 'code' and cell.source.startswith('%%writefile '):
        first, source = cell.source.split('\n', 1)
        assert source == (root / first.split()[1]).read_text(encoding='utf-8')
        assert 'TODO' not in source
        written += 1
assert written == 6
assert (10_000_000 + 511) // 512 == 19532
assert ((10000 + 31) // 32, (10000 + 15) // 16) == (313, 625)

def kernel(filename, name):
    s = (root / filename).read_text()
    return s[s.index('__global__'):s.index('\nvoid random_ints')].replace('__global__ ', '')

work = root / '.validation'
work.mkdir(exist_ok=True)
source = r'''
#include <cassert>
#include <cstdio>
#include <vector>
struct D { int x, y; } blockIdx, blockDim, threadIdx;
'''
source += kernel('ex5_vecmul.cu', 'vecMul') + kernel('ex6_matrix.cu', 'mulElem')
source += r'''
int main() {
    for (int n : {1, 511, 512, 513, 1025}) {
        std::vector<int> a(n+1, 7), b(n+1, 9), c(n+1, -1);
        blockDim={512,1};
        for (int block=0; block<(n+511)/512; ++block)
            for (int t=0; t<512; ++t) {
                blockIdx={block,0}; threadIdx={t,0};
                vecMul(a.data(),b.data(),c.data(),n);
            }
        for (int i=0; i<n; ++i) assert(c[i]==63);
        assert(c[n]==-1);
    }
    for (int rows : {1,16,17,33}) for (int cols : {1,31,32,33,65}) {
        int n=rows*cols;
        std::vector<int> a(n+1), b(n+1), c(n+1,-1);
        for (int i=0;i<=n;++i) { a[i]=i%100; b[i]=(i*7)%100; }
        blockDim={32,16};
        for (int y=0;y<(rows+15)/16;++y) for (int x=0;x<(cols+31)/32;++x)
            for (int ty=0;ty<16;++ty) for (int tx=0;tx<32;++tx) {
                blockIdx={x,y}; threadIdx={tx,ty};
                mulElem(a.data(),b.data(),c.data(),rows,cols);
            }
        for (int i=0;i<n;++i) assert(c[i]==a[i]*b[i]);
        assert(c[n]==-1);
    }
    puts("PASS: vector and matrix indexing, products and bounds guards");
}
'''
(work / 'kernel_check.cpp').write_text(source)
subprocess.run(['g++', '-std=c++11', '-Wall', '-Wextra', str(work / 'kernel_check.cpp'), '-o', str(work / 'kernel_check.exe')], check=True)
subprocess.run([str(work / 'kernel_check.exe')], check=True)
print('PASS: notebook schema, six matching source cells, required launch dimensions')
