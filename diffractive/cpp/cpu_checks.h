#ifndef CHECKS_H
#define CHECKS_H

#define CHECK_CPU(x) TORCH_CHECK(!x.is_cuda(), #x " must be a CPU tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CPU(x); CHECK_CONTIGUOUS(x)

#endif
