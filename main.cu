#include <random>
#include <assert.h>
#include <chrono>
#include <iostream>

using real = float;

#define DEBUG

const real tau_v = 20.;
const real tau_exc = 5.;
const real tau_inh = 10.;
const real v_thresh = -50.;
const real v_reset = -60.;
const real v_rest = -49.;
const real wgt_exc = 60.*.27/5;
const real wgt_inh = -20*4.5/10;

const real ts = .1; // ms
// refractory period is 5ms
// gotta manually divide 5 / ts here because of FP error...
const unsigned char refractory_cycles = 50;

// Simulation parameters
const size_t N = 100000;
const size_t delay = 8;
const size_t max_conns_per_neuron = 1000;
const real seconds = 10.;
const size_t num_iterations = seconds / (ts * 1e-3) / delay;
const double resulting_sparsity = 1. * max_conns_per_neuron / N;
const size_t N_exc = N * 4 / 5;

// gpu optimization params
const size_t threads_per_block = 1000;
const size_t n_blocks = N / threads_per_block;

struct Injection {
  real exc, inh;
  Injection(real a, real b): exc(a), inh(b) {}
  Injection() : exc(0.), inh(0.) {}
};

struct Connection {
  unsigned int idx;
  real wgt;
  Connection(unsigned int d, real w) : idx(d), wgt(w) {}
  Connection() : idx(0), wgt(0.) {}
};

const size_t bank_size = N * delay;

#define injection(polarity, delay_idx, neuron_idx) (bank_injections[polarity * bank_size + delay_idx * N + neuron_idx])
#define connection(neuron_idx, synapse_idx) (connections[neuron_idx * max_conns_per_neuron + synapse_idx])
unsigned char __ping_var = 0;
#include <stdio.h>
#define ping() fprintf(stderr, "ping %d\n", __ping_var++); fflush(stderr);

__global__
void iterate(real * v, real * ge, real * gi, unsigned char * refrac, Connection * connections, Injection * bank_injections, bool polarity
#ifdef DEBUG
    , int *nspikes
#endif
    ) {
  size_t idx = blockDim.x * blockIdx.x + threadIdx.x;
  __shared__ bool spikes[threads_per_block];

  for (unsigned char delay_idx = 0; delay_idx < delay; ++delay_idx) {
    real dv = (ge[idx] + gi[idx] - (v[idx] - v_rest)) / tau_v;
    real dge = -ge[idx] / tau_exc;
    real dgi = -gi[idx] / tau_inh;

    if (refrac[idx]) {
      --refrac[idx];
      dv = 0.;
    }

    // read once to local register
    real v_ = v[idx] + dv * ts;
    Injection *inj = &injection(polarity, delay_idx, idx);
    ge[idx] += inj->exc + dge * ts;
    gi[idx] += inj->inh + dgi * ts;
    inj->exc = 0.;
    inj->inh = 0.;

    bool spiked = v_ > v_thresh;
    spikes[threadIdx.x] = spiked;

    if (spiked) {
      v_ = v_reset;
      refrac[idx] = refractory_cycles;
    }

    v[idx] = v_;

    __syncthreads();
    for (unsigned int lidx = 0; lidx < blockDim.x; ++lidx) {
      // this actually isn't that bad because either all threads take it or all don't
      if (spikes[lidx]) {
#ifdef DEBUG
        if (threadIdx.x == 0)
          atomicAdd(nspikes, 1);
#endif
        size_t tidx = blockDim.x * blockIdx.x + lidx;
        Connection c = connection(tidx, threadIdx.x);
        real wgt = c.wgt;
        // We read from polarity, and write to ~polarity to avoid race conditions across blocks
        if (c.wgt > 0)
          atomicAdd(&injection(!polarity, delay_idx, c.idx).exc, wgt);
        else
          atomicAdd(&injection(!polarity, delay_idx, c.idx).inh, wgt);
      }
    }
  }
}

int main() {
  std::default_random_engine gen;
  std::uniform_real_distribution<> voltage_dist(v_reset, v_thresh);
  std::poisson_distribution<> connection_dist(N / max_conns_per_neuron);
//  std::uniform_real_distribution<> unit_dist(0., 1.);

  real * neuron_v;
  real * cuda_neuron_v;
  real * neuron_ge;
  real * cuda_neuron_ge;
  real * neuron_gi;
  real * cuda_neuron_gi;
  unsigned char * neuron_ref_cycles_rem;
  unsigned char * cuda_neuron_ref_cycles_rem;
  Connection * connections;
  Connection * cuda_connections;

  Injection * bank_injections;
  Injection * cuda_bank_injections;


  // allocate

  neuron_v = new real[N];
  assert(cudaSuccess == cudaMalloc(&cuda_neuron_v, sizeof(real) * N));
  neuron_ge = new real[N];
  assert(cudaSuccess == cudaMalloc(&cuda_neuron_ge, sizeof(real) * N));
  neuron_gi = new real[N];
  assert(cudaSuccess == cudaMalloc(&cuda_neuron_gi, sizeof(real) * N));
  neuron_ref_cycles_rem = new unsigned char[N];
  assert(cudaSuccess == cudaMalloc(&cuda_neuron_ref_cycles_rem, sizeof(unsigned char) * N));
  connections = new Connection[max_conns_per_neuron * N];
  assert(cudaSuccess == cudaMalloc(&cuda_connections, sizeof(Connection) * max_conns_per_neuron * N));

  bank_injections = new Injection[2 * N * delay];
  assert(cudaSuccess == cudaMalloc(&cuda_bank_injections, sizeof(Injection) * 2 * N * delay));

  ping();

  // initialize
  for (size_t i = 0; i < N; ++i) {
    neuron_v[i] = voltage_dist(gen);
    neuron_ge[i] = neuron_gi[i] = 0.;
    neuron_ref_cycles_rem[i] = 0;
    size_t synapse_idx = connection_dist(gen) - 1;
    for (unsigned conn_idx = 0; conn_idx < 1000 && synapse_idx < N; ++conn_idx) {
      real wgt = (i < N_exc) ? wgt_exc : wgt_inh;
      connection(i, conn_idx) = Connection(synapse_idx, wgt);
      synapse_idx += connection_dist(gen);
    }
  }

  ping();

  // copy to GPU
  assert(cudaSuccess == cudaMemcpy(cuda_neuron_v, neuron_v, sizeof(real) * N, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(cuda_neuron_ge, neuron_ge, sizeof(real) * N, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(cuda_neuron_gi, neuron_gi, sizeof(real) * N, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(cuda_neuron_ref_cycles_rem, neuron_ref_cycles_rem, sizeof(unsigned char) * N, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(cuda_connections, connections, sizeof(Connection) * N * max_conns_per_neuron, cudaMemcpyHostToDevice));
  assert(cudaSuccess == cudaMemcpy(cuda_bank_injections, bank_injections, sizeof(Injection) * 2 * N * delay, cudaMemcpyHostToDevice));
  ping();

#ifdef DEBUG
  int nspikes = 0;
  int * cuda_nspikes;
  assert(cudaSuccess == cudaMalloc(&cuda_nspikes, sizeof(int)));
  assert(cudaSuccess == cudaMemcpy(cuda_nspikes, &nspikes, sizeof(int), cudaMemcpyHostToDevice));
#endif

  // run
  bool polarity = false;
  std::cout << "begin!" << std::endl;
  auto t1 = std::chrono::high_resolution_clock::now();
  for (size_t it = 0; it < num_iterations; ++it) {
    iterate<<<n_blocks, threads_per_block>>>(cuda_neuron_v, cuda_neuron_ge, cuda_neuron_gi, cuda_neuron_ref_cycles_rem, cuda_connections, cuda_bank_injections, polarity
#ifdef DEBUG
    , cuda_nspikes
#endif
    );
    polarity = !polarity;
  }
  auto t2 = std::chrono::high_resolution_clock::now();
  auto diff = (t2 - t1);
  std::cout << "Time Elapsed: " << (diff.count() / 1e9) << std::endl;
#ifdef DEBUG
  cudaMemcpy(&nspikes, cuda_nspikes, sizeof(int), cudaMemcpyDeviceToHost);
  std::cout << "Firing Rate: " << (1. * nspikes / N / seconds) << "Hz" << std::endl;
#endif
}
