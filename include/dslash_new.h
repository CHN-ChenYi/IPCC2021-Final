/**
** @file:  dslash_new.h
** @brief: define Dslash and Dlash-dagger operaters, 
           i.e.  dest = M(U) src &  dest = M(U)^\dagger src. 
           Note: even-odd precondition trick to be used.
**/

#ifndef LATTICE_DSLASH_NEW_H
#define LATTICE_DSLASH_NEW_H

#include "lattice_fermion.h"
#include "lattice_gauge.h"
#include "operator.h"
#include "operator_mpi.h"

// void DslashEENew(lattice_fermion &src, lattice_fermion &dest, const double mass);
// void DslashOONew(lattice_fermion &src, lattice_fermion &dest, const double mass);
void DslashEEOONew(lattice_fermion &src, lattice_fermion &dest, const double mass);
void DslashEONew(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag);
void DslashOENew(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag);
void DslashoffdNew(lattice_fermion &src, lattice_fermion &dest, lattice_gauge &U, const bool dag,
                   int cb);
#endif
