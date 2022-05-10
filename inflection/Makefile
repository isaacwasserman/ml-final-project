UNAME := $(shell uname)

ifeq ($(UNAME), Darwin)
CFLAGS = -Ofast
endif

ifeq ($(UNAME), Linux)
CFLAGS = -O3
endif

all: libalign.so

libalign.so: align.c
	gcc -O3 -Wall -Wextra -shared -fPIC align.c -o libalign.so

clean:
	/bin/rm libalign.so *.pyc



