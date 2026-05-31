CC = gcc
CFLAGS = -O3 -march=native -pthread
all: src/wdb_kernel src/wdb_kernel_count
src/wdb_kernel: src/wdb_kernel.c
	$(CC) $(CFLAGS) -o $@ $<
src/wdb_kernel_count: src/wdb_kernel_count.c
	$(CC) $(CFLAGS) -o $@ $<
clean:
	rm -f src/wdb_kernel src/wdb_kernel_count
