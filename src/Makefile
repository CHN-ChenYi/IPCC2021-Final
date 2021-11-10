CC       = mpicc
CXX      = mpiicpc

CXX_FLAGS = -fPIC -I../include -std=c++11 -Ofast -ffast-math -march=core-avx2 -fomit-frame-pointer -DNO_MPI_IO
#CXX_FLAGS = -fPIC -I../include -DVERBOSE_SIMPLE
#CXX_FLAGS = -fPIC -I../include -DDEBUG -DLIME -NO_MPI_IO -g
LINK_FLAGS = -lmpi

SLIB = liblattice.so
OBJS= dslash.o dslash_new.o lattice_fermion.o lattice_gauge.o invert.o  check.o load_gauge.o
TARGET= main

all: ${TARGET}

.SUFFIXES: .cpp .c
.cpp.o:
	${CXX} ${CXX_FLAGS} -o $@ -c $<

${SLIB}: ${OBJS}
	${CXX} --shared $^ -o $@

${TARGET}: main.cpp ${SLIB}
	${CXX} ${CXX_FLAGS} -Wl,-rpath=./ ${LINK_FLAGS} -o $@  $< -L./ -llattice

fmt:
	@#cd .. && git ls-files -- '*.cpp' '*.h' | xargs clang-format -i -style=file
	find .. -name "*.cpp" -or -name "*.h" | xargs clang-format -i -style=file

run: all
	sbatch sub.sh
	squeue

clean:
	rm -rf ${SLIB} ${OBJS} ${TARGET}
