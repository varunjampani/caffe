#ifndef CAFFE_PERMUTOHEDRAL_HPP_
#define CAFFE_PERMUTOHEDRAL_HPP_

#include "caffe/common.hpp"

#pragma once
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <cstdio>
#include <cmath>
#include <vector>
#include <memory>
#include <array>
#include <tuple>
 

namespace caffe {

    /************************************************/
    /***                Hash Table                ***/
    /************************************************/

    class HashTable{
      private:
        // Don't copy!
        HashTable( const HashTable & o ): key_size_ ( o.key_size_ ), filled_(0), capacity_(o.capacity_) {
          table_ = new int[ capacity_ ];
          keys_ = new short[ (capacity_/2+10)*key_size_ ];
          memset( table_, -1, capacity_*sizeof(int) );
        }

      private:
        size_t key_size_, filled_, capacity_;
        short * keys_;
        int * table_;
        void grow(){
          // Swap out the old memory
          short * old_keys = keys_;
          int * old_table = table_;
          int old_capacity = capacity_;
          capacity_ *= 2;
          // Allocate the new memory
          keys_ = new short[ (old_capacity+10)*key_size_ ];
          table_ = new int[ capacity_ ];
          memset( table_, -1, capacity_*sizeof(int) );
          memcpy( keys_, old_keys, filled_*key_size_*sizeof(short) );

          // Reinsert each element
          for( int i=0; i<old_capacity; i++ )
            if (old_table[i] >= 0){
              int e = old_table[i];
              size_t h = hash( old_keys+(getKey(e)-keys_) ) % capacity_;
              for(; table_[h] >= 0; h = h<capacity_-1 ? h+1 : 0);
              table_[h] = e;
            }

          delete [] old_keys;
          delete [] old_table;
        }
        size_t hash( const short * k ) {
          size_t r = 0;
          for( size_t i=0; i<key_size_; i++ ){
            r += k[i];
            r *= 1664525;
          }
          return r;
        }
      public:
        explicit HashTable( int key_size, int n_elements )
          : key_size_ ( key_size ), filled_(0), capacity_(2*n_elements) {
          table_ = new int[ capacity_ ];
          keys_ = new short[ (capacity_/2+10)*key_size_ ];
          memset( table_, -1, capacity_*sizeof(int) );
        }
        ~HashTable() {
          delete [] keys_;
          delete [] table_;
        }
        int size() const {
          return filled_;
        }
        void reset() {
          filled_ = 0;
          memset( table_, -1, capacity_*sizeof(int) );
        }
        int find( const short * k, bool create = false ){
          if (2*filled_ >= capacity_) grow();
          // Get the hash value
          size_t h = hash( k ) % capacity_;
          // Find the element with he right key, using linear probing
          while(1){
            int e = table_[h];
            if (e==-1){
              if (create){
                // Insert a new key and return the new id
                for( size_t i=0; i<key_size_; i++ )
                  keys_[ filled_*key_size_+i ] = k[i];
                return table_[h] = filled_++;
              }
              else
                return -1;
            }
            // Check if the current key is The One
            bool good = true;
            for( size_t i=0; i<key_size_ && good; i++ )
              if (keys_[ e*key_size_+i ] != k[i])
                good = false;
            if (good)
              return e;
            // Continue searching
            h++;
            if (h==capacity_) h = 0;
          }
        }
        const short * getKey( int i ) const{
          return keys_+i*key_size_;
        }

    };

    /************************************************/
    /***          Permutohedral Lattice           ***/
    /************************************************/

    template <typename T>
    class Permutohedral
    {
      public:
        typedef T value_type;

      private:
        // don't copy
        Permutohedral(const Permutohedral& rhs);

        typedef typename std::array<float, 7> filter_type;
        const filter_type filter_;

        typedef std::array<int, (std::tuple_size<filter_type>::value
            - 1) * 2> Neighbors;

        float alpha_;

        int N_, d_;
        int M_;
        std::vector<value_type> barycentric_;
        std::vector<short> rank_;
        std::vector<int> offset_;
        std::vector<Neighbors> blur_neighbors_;

      public:
        Permutohedral()
          //: filter_(filter_type {{2.0 / 2.0, 1.0 / 2.0}}) {
          //: filter_(filter_type {{70, 56, 28, 8, 1}}) {
          : filter_(filter_type {{924, 729, 495, 220, 66, 12, 1}}) {
        }

        void init(const value_type* feature, int data_count, int
            feature_size) {
          N_ = data_count;
          d_ = feature_size;

          // initialize splat correction term -- this depends on the filter
          // this ensures that a blurred delta peak with weight 1 still is a
          // blurred delta peak with weight 1
          alpha_ = powf(filter_[0], d_+1);
          for (int n = 1; n < static_cast<int>(filter_.size()); n++) {
            alpha_ += 2 * powf(filter_[n], d_+1);
          }
          alpha_ = 1 / alpha_;

          // allocate enough storage
          barycentric_.resize(static_cast<std::size_t>((d_+1) * N_));
          rank_.resize((d_+1) * N_);
          offset_.resize((d_+1) * N_);

          // Compute the lattice coordinates for each feature [there is going to be
          // a lot of magic here
          HashTable hash_table( d_, N_*(d_+1) );

          // Allocate the local memory
          std::vector<float> scale_factor(d_);
          std::vector<value_type> elevated(d_+1);
          std::vector<float> rem0(d_+1);
          std::vector<value_type> barycentric(d_+2);
          std::vector<short> canonical((d_+1)*(d_+1));
          std::vector<short> key(d_+1);

          // Compute the canonical simplex
          for( int i=0; i<=d_; i++ ){
            for( int j=0; j<=d_-i; j++ )
              canonical[i*(d_+1)+j] = i;
            for( int j=d_-i+1; j<=d_; j++ )
              canonical[i*(d_+1)+j] = i - (d_+1);
          }

          // Expected standard deviation of our filter (p.6 in [Adams etal 2010])
          float inv_std_dev = sqrt(1.0 / 6.0
              + (std::tuple_size<filter_type>::value - 1) / 2.0)*(d_+1);
          // Compute the diagonal part of E (p.5 in [Adams etal 2010])
          for( int i=0; i<d_; i++ )
            scale_factor[i] = 1.0 / sqrt( (i+2)*(i+1) ) * inv_std_dev;

          // Compute the simplex each feature lies in
          for( int k=0; k<N_; k++ ){
            // Elevate the feature ( y = Ep, see p.5 in [Adams etal 2010])
            const value_type * f = feature + k*feature_size;

            // sm contains the sum of 1..n of our faeture vector
            value_type sm(0);
            for( int j=d_; j>0; j-- ){
              value_type cf = f[j-1]*scale_factor[j-1];
              elevated[j] = sm - j*cf;
              sm += cf;
            }
            elevated[0] = sm;

            // Find the closest 0-colored simplex through rounding
            float down_factor = 1.0f / (d_+1);
            float up_factor = (d_+1);
            int sum = 0;
            for( int i=0; i<=d_; i++ ){
              int rd = round( down_factor * static_cast<float>(elevated[i]) );
              rem0[i] = rd*up_factor;
              sum += rd;
            }

            // Find the simplex we are in and store it in rank (where rank
            // describes what position coorinate i has in the sorted order of the
            // features values)
            short* rank = rank_.data() + (d_+1)*k;
            for( int i=0; i<d_; i++ ){
              double di = static_cast<float>(elevated[i]) - rem0[i];
              for( int j=i+1; j<=d_; j++ )
                if ( di < static_cast<float>(elevated[j]) - rem0[j])
                  rank[i]++;
                else
                  rank[j]++;
            }

            // If the point doesn't lie on the plane (sum != 0) bring it back
            for( int i=0; i<=d_; i++ ){
              rank[i] += sum;
              if ( rank[i] < 0 ){
                rank[i] += d_+1;
                rem0[i] += d_+1;
              }
              else if ( rank[i] > d_ ){
                rank[i] -= d_+1;
                rem0[i] -= d_+1;
              }
            }

            // Compute the barycentric coordinates (p.10 in [Adams etal 2010])
            for( int i=0; i<=d_+1; i++ )
              barycentric[i] = 0;
            for( int i=0; i<=d_; i++ ){
              value_type v = (elevated[i] - rem0[i])*down_factor;

              if (d_-rank[i] < 0 || d_-rank[i]+1 >= d_+2)
                throw std::runtime_error("Permutohedral: rank access error");

              //assert(d_-rank[i]   >= 0);
              //assert(d_-rank[i]+1 <  d_+2);
              barycentric[d_-rank[i]  ] += v;
              barycentric[d_-rank[i]+1] -= v;
            }
            // Wrap around
            barycentric[0] += 1.0 + barycentric[d_+1];

            // Compute all vertices and their offset
            std::vector<short> neighborKeyUp(d_ + 1);
            std::vector<short> neighborKeyDown(d_ + 1);
            for( int remainder=0; remainder<=d_; remainder++ ){
              for( int i=0; i<d_; i++ )
                key[i] = rem0[i] + canonical[ remainder*(d_+1) + rank[i] ];
              assert(k*(d_+1)+remainder < (d_+1) * N_);
              offset_[ k*(d_+1)+remainder ] = hash_table.find( key.data(), true );
              barycentric_[k*(d_+1)+remainder] = barycentric[remainder];

              // add blur neighbors, too
              /*for( int j = 0; j <= d_; j++ ) {
                for (int k = 0; k < d_; k++) {
                  neighborKeyUp[k] = key[k];
                  neighborKeyDown[k] = key[k];
                }
                for (int n = 0; n < filter_.size() - 1; n++) {
                  for (int k = 0; k < d_; k++) {
                    neighborKeyUp[k] += 1;
                    neighborKeyDown[k] += -1;
                  }
                  neighborKeyUp[j] += -1 - d_;
                  neighborKeyDown[j] += 1 + d_;

                  hash_table.find(neighborKeyUp.data(), true);
                  hash_table.find(neighborKeyDown.data(), true);
                }
              }*/
            }

          }

          // Find the Neighbors of each lattice point

          // Get the number of vertices in the lattice
          M_ = hash_table.size();

          //std::cout << "M_ " << M_ << std::endl;

          // Create the neighborhood structure
          blur_neighbors_.resize((d_+1)*M_);

          std::vector<short> n1(d_+1);
          std::vector<short> n2(d_+1);

          // For each of d+1 axes,
          for( int j = 0; j <= d_; j++ ) {
            for( int i=0; i<M_; i++ ){

              Neighbors& neighbors = blur_neighbors_[j*M_+i];

              // up
              const short * key = hash_table.getKey( i );
              std::vector<short> neighborKeyUp(d_ + 1);
              std::vector<short> neighborKeyDown(d_ + 1);
              // init with the current location
              // remember: keys are not evaluated on their last entry (d_)
              // because it is implicitly defined
              for (int k = 0; k < d_; k++) {
                neighborKeyUp[k] = key[k];
                neighborKeyDown[k] = key[k];
              }
              for (int n = 0; n < static_cast<int>(filter_.size()) - 1; n++) {
                for (int k = 0; k < d_; k++) {
                  neighborKeyUp[k] += 1;
                  neighborKeyDown[k] += -1;
                }
                neighborKeyUp[j] += -1 - d_;
                neighborKeyDown[j] += 1 + d_;

                neighbors[2*n] = hash_table.find(neighborKeyUp.data());
                neighbors[2*n+1] = hash_table.find(neighborKeyDown.data());
              }
            }
          }
        }

        void slice(const std::vector<value_type>& data, const std::size_t
            value_size, const std::size_t out_offset, const std::size_t
            out_size, value_type* sliced) const {
          for( int i=0; i<static_cast<int>(out_size); i++ ){
            for( int k=0; k<static_cast<int>(value_size); k++)
              sliced[i*value_size+k] = 0;

            for( int j=0; j<=d_; j++ ){
              int o = offset_[(out_offset+i)*(d_+1)+j]+1;
              value_type w = barycentric_[(out_offset+i)*(d_+1)+j];
              for( int k=0; k<static_cast<int>(value_size); k++ )
                sliced[i*value_size+k] += w * data[o*value_size+k] * alpha_;
            }
          }
        }

        template <typename V>
        void blur(V&& splatted, const std::size_t value_size,
            std::vector<value_type>& blurred) const {
          // allow move
          blurred = std::forward<V>(splatted);
          std::vector<value_type> new_blurred(blurred.size());

          for( int j=0; j<=d_; j++ ){
            for( int i=0; i<M_; i++ ){
              assert((i+1) * value_size < blurred.size());
              value_type * old_val = blurred.data() + (i+1)*value_size;
              value_type * new_val = new_blurred.data() + (i+1)*value_size;

              const Neighbors& neighbors = blur_neighbors_[j*M_+i];

              // initialize with weighted center vertex
              for (int k=0; k<static_cast<int>(value_size); k++) {
                new_val[k] = old_val[k] * filter_[0];
              }

              // now iterate over all vertex neighbors and blur
              for (int n=0; n<static_cast<int>(filter_.size())-1; n++) {
                int neighbor_up = neighbors[2*n] + 1;
                int neighbor_down = neighbors[2*n + 1] + 1;
                assert(neighbor_up < M_+1 && neighbor_down < M_+1);
                assert(neighbor_up >= 0 && neighbor_down >= 0);

                // get the pointer to the data
                value_type * up_val = blurred.data() + neighbor_up*value_size;
                value_type * down_val = blurred.data() + neighbor_down*value_size;

                for (int k=0; k<static_cast<int>(value_size); k++) {
                  new_val[k] += (up_val[k] + down_val[k]) * filter_[n+1];
                }
              }
            }

            new_blurred.swap(blurred);
          }
        }

        void splat(const value_type* blurred, const std::size_t
            value_size, const std::size_t in_offset, const std::size_t in_size,
            std::vector<value_type>& splatted) const {
          for (int i=0; i<static_cast<int>(in_size); i++) {
            for ( int j=0; j<=d_; j++) {
              int o = offset_[(in_offset+i)*(d_+1)+j]+1;
              const value_type& w = barycentric_[(in_offset+i)*(d_+1)+j];
              for (int k=0; k<static_cast<int>(value_size); k++)
                splatted[o*value_size+k] += w * blurred[i*value_size+k];
            }
          }
        }

        void
        compute(const value_type* in, int value_size, int
            in_offset, int out_offset, int in_size, int out_size,
            value_type* out)
        const {
          std::vector<value_type> tmp((M_+2)*value_size);
          splat(in, value_size, in_offset, in_size, tmp);

          blur(tmp, value_size, tmp);

          slice(tmp, value_size, out_offset, out_size, out);
        }
    };
  }
}  // namespace caffe

#endif
