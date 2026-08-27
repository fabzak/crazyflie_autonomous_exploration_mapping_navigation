CC      ?= gcc
WARN     = -Wall -Wextra -Werror -Wshadow -Wconversion -Wsign-conversion \
           -Wpointer-arith -Wcast-qual -Wstrict-prototypes -Wmissing-prototypes \
           -Wredundant-decls -Wundef -Wdouble-promotion
CFLAGS  ?= -std=c11 -O2 -g $(WARN)
SAN      = -fsanitize=address,undefined -fno-omit-frame-pointer

all: selftest trace_runner

datum_lifecycle.o: datum_lifecycle.c datum_lifecycle.h
	$(CC) $(CFLAGS) -c $< -o $@

selftest: lifecycle_selftest.c datum_lifecycle.o datum_lifecycle.h
	$(CC) $(CFLAGS) lifecycle_selftest.c datum_lifecycle.o -o $@ -lm

trace_runner: trace_runner.c datum_lifecycle.o datum_lifecycle.h
	$(CC) $(CFLAGS) trace_runner.c datum_lifecycle.o -o $@ -lm

san: lifecycle_selftest.c datum_lifecycle.c datum_lifecycle.h
	$(CC) $(CFLAGS) $(SAN) lifecycle_selftest.c datum_lifecycle.c -o selftest_san -lm

clean:
	rm -f *.o selftest trace_runner selftest_san

.PHONY: all san clean
