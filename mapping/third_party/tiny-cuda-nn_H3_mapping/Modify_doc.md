# Modified part

1. Add HASH_TYPE MultiPrime

   Prime list is in  grid.h 

   ```c++
   __constant__ uint32_t factors[40][7] 
   ```

2. Remove the offset 0.5 in common_device.h 

   The 0.5 offset is used in the smooth interp, but we do not need it.

   ```c++
   template <typename F, typename FPRIME, typename FPRIMEPRIME>
   __device__ inline void
   pos_fract(const float input, float *pos, float *pos_derivative,
             float *pos_2nd_derivative, uint32_t *pos_grid, float scale,
             F interpolation_fun, FPRIME interpolation_fun_derivative,
             FPRIMEPRIME interpolation_fun_2nd_derivative) {
   	// The offset of 0.5 causes different scales to be staggered with respect to each other, thus
   	// preventing spurious alignment of fractional coordinates upon integer scales (or powers thereof).
   	// This is mentioned in Appendix A of the "Instant Neural Graphics Primitives" paper.
   	// The offset can cause wraparound indexing in dense grids, which didn't negatively impact
   	// the approximation quality in any of our tests.
   //  *pos = fmaf(scale, input, 0.5f); // raw
     *pos = fmaf(scale, input, 0.0f); // jcx_modify
     float tmp = floorf(*pos);
     *pos_grid = (uint32_t)(int)tmp;
     *pos -= (float)tmp;
     *pos_2nd_derivative = interpolation_fun_2nd_derivative(*pos);
     *pos_derivative = interpolation_fun_derivative(*pos);
     *pos = interpolation_fun(*pos);
   }
   
   ```

3. Always use hash index

   In the original version, if the number of grids is lower than the hashmap_size, it will choose a "Dense" mode to allocate the feature. 

   (1)  in grid.h

   ```c++
   GridEncodingTemplated
   ...
   else if (grid_type == GridType::Hash) {
           // If hash table needs fewer params than dense, then use fewer and rely
           // on the hash.
   //        params_in_level = std::min(params_in_level, (1u << log2_hashmap_size)); // raw
           params_in_level = 1u << log2_hashmap_size; // jcx_modify
   ```

   (2) in common_device.h

   ```c++
   template <uint32_t N_DIMS, HashType HASH_TYPE>
   __device__ uint32_t grid_index(const GridType grid_type,
                                  const uint32_t hashmap_size,
                                  const uint32_t grid_resolution,
                                  const uvec<N_DIMS> &pos_grid,
                                  const uint32_t* primes_ = nullptr) {
     // uint32_t stride = 1;
     uint32_t index = 0;
   
     // raw
     // // The second part of the loop condition is needed to avoid integer overflows
     //  // in finer levels.
     // TCNN_PRAGMA_UNROLL
     // for (uint32_t dim = 0; dim < N_DIMS && stride <= hashmap_size; ++dim) {
     //   index += pos_grid[dim] * stride;
     //   stride *= grid_resolution;
     // }
   
     // if (grid_type == GridType::Hash && hashmap_size < stride) {
     //   index = grid_hash<N_DIMS, HASH_TYPE>(pos_grid, primes_);
     // }
   
     // jcx_modify
     if (grid_type == GridType::Hash) {
       index = grid_hash<N_DIMS, HASH_TYPE>(pos_grid,primes_);
     }
     else {
     	   uint32_t stride = 1;
   	   TCNN_PRAGMA_UNROLL
   	   for (uint32_t dim = 0; dim < N_DIMS && stride <= hashmap_size; ++dim) {
   		 index += pos_grid[dim] * stride;
   		 stride *= grid_resolution;
   	   }
   
   	   if (grid_type == GridType::Hash && hashmap_size < stride) {
   		 index = grid_hash<N_DIMS, HASH_TYPE>(pos_grid, primes_);
   	   }
     }
   //  printf("index: %u\n",index);
   //  printf("hashmap_size: %u\n",hashmap_size);
   //  printf("index hashmap_size: %u\n",index % hashmap_size);
     return index % hashmap_size;
   }
   ```

   

