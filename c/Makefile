# C 移植的构建脚本
#
# 如果你的环境没有 make，直接编译即可（Windows/MinGW 与 Linux/macOS 通用）：
#   gcc -std=c17 -O2 -Wall -Wextra -o demo demo.c ai.c -lm
#   gcc -std=c17 -O2 -Wall -Wextra -o compare_python compare_python.c ai.c -lm
#   gcc -std=c17 -O2 -Wall -Wextra -o bench bench.c ai.c -lm

CC      ?= gcc
CFLAGS  ?= -std=c17 -O2 -Wall -Wextra
LDLIBS  ?= -lm

BINS = demo compare_python bench

all: $(BINS)

demo: demo.c ai.c ai.h
	$(CC) $(CFLAGS) -o $@ demo.c ai.c $(LDLIBS)

compare_python: compare_python.c ai.c ai.h
	$(CC) $(CFLAGS) -o $@ compare_python.c ai.c $(LDLIBS)

bench: bench.c ai.c ai.h
	$(CC) $(CFLAGS) -o $@ bench.c ai.c $(LDLIBS)

# 与 Python 做数值等价性比对
check: compare_python
	python ../tools/c_reference.py reference.txt
	./compare_python reference.txt

clean:
	rm -f $(BINS) $(BINS:=.exe) *.o reference.txt

.PHONY: all check clean
