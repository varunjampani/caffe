#ifndef _DIFF_BLUR_798D92894727_
#define _DIFF_BLUR_798D92894727_

#include <autodiff_rvs_op/autodiff.h>

#include <memory>
#include <cstddef>
#include <stdexcept>

namespace densecrf {
  namespace internal {
    template <typename TValue, typename TBlur>
    class DiffBlur {
      public:
        typedef TValue value_type;
        typedef TBlur blur_type;

      private:
        typedef typename ad::Op<value_type, 1>::vector_type vector_type;
        typedef typename ad::Op<value_type, 1>::dim_type dim_type;

      public:
        DiffBlur() : blurData_() {
        }

        void init(const std::shared_ptr<ad::Op<value_type, 1>>& featureOp,
            const std::size_t featureSize) {
          const vector_type& feature = featureOp->value();

          if ((feature.size() % featureSize) != 0)
            throw std::runtime_error(
                "DiffBlur::init: feature vector has to be a multiple of feature size");

          blurData_ = std::make_shared<BlurData>(featureOp, feature.size() /
              featureSize, featureSize);

          blurData_->blur_.init(feature.data(), blurData_->dataCount_,
              blurData_->featureSize_);

        }

        std::shared_ptr<ad::Op<value_type, 1>>
        compute(const std::shared_ptr<ad::Op<value_type, 1>>& in, int
            valueSize, int inOffset=0, int outOffset=0, int outSize = -1)
        const {
          if (outSize == -1) outSize = blurData_->dataCount_ - outOffset;

          return std::make_shared<BlurOp>(blurData_, in,
              valueSize, inOffset, outOffset, outSize);
        }

      private:
        struct BlurData {
          BlurData(const std::shared_ptr<ad::Op<value_type, 1>>& featureOp,
              std::size_t dataCount, std::size_t featureSize)
          : featureOp_(featureOp), blur_(), dataCount_(dataCount),
            featureSize_(featureSize)
          {
          }

          std::shared_ptr<ad::Op<value_type, 1>> featureOp_;
          blur_type blur_;

          const std::size_t dataCount_;
          const std::size_t featureSize_;
        };

        class BlurOp : public ad::RefOp<value_type, 1> {
          public:
            BlurOp(const std::shared_ptr<BlurData>& blurData,
                const std::shared_ptr<ad::Op<value_type, 1>>& dataOp,
                int valueSize, int inOffset, int outOffset, int outSize)
              : blurData_(blurData), dataOp_(dataOp),
                blurred_(),
                valueSize_(valueSize),
                inOffset_(inOffset),
                outOffset_(outOffset),
                inSize_(dataOp->value().size() / valueSize),
                outSize_(outSize)
            {
              if ((dataOp->value().size() % valueSize) != 0)
                throw std::runtime_error(
                    "DiffBlur::BlurOp: data vector has to be a multiple of value size");

              dataOp_->reference();
              //blurData_->featureOp_->reference();

              blurred_.resize(dim_type {{outSize_ * valueSize_}});

              blurData_->blur_.compute(dataOp->value().data(), valueSize_,
                  inOffset_, outOffset_, inSize_, outSize_, blurred_.data());
            }

            virtual
            const vector_type& value() const {
              return blurred_;
            }

          private:
            virtual void reverseImpl(const vector_type& grad) {
              /*vector_type featureGrad((dim_type {{
                    static_cast<std::size_t>(blurData_->dataCount_ *
                      blurData_->featureSize_)}}));*/
              vector_type dataGrad((dim_type {{
                    static_cast<std::size_t>(inSize_ *
                      valueSize_)}}));

              /*const vector_type& feature = blurData_->featureOp_->value();
              const vector_type& data = dataOp_->value();

              // feature gradient -- out data
              // prepare the blur data
              const int outGradientDataDim = blurData_->featureSize_ *
                valueSize_ + valueSize_;
              std::vector<value_type> outGradData(outGradientDataDim * outSize_);

              const int outGradOffset = blurData_->featureSize_ * valueSize_;

              value_type* outGradP = outGradData.data();
              const value_type* featureP = &feature[outOffset_ * blurData_->featureSize_];
              const value_type* gradPCurrent = grad.data();
              for (std::size_t i = 0; i < outSize_; ++i) {
                const value_type* gradP;
                for (std::size_t d = 0; d < blurData_->featureSize_; ++d) {
                  gradP = gradPCurrent;
                  for (std::size_t e = 0; e < valueSize_; ++e) {
                    // grad + feature
                    (*outGradP++) = (*gradP++) * (*featureP);
                  }

                  ++featureP;
                }

                gradP = gradPCurrent;
                for (std::size_t e = 0; e < valueSize_; ++e) {
                  (*outGradP++) = (*gradP++);
                }

                gradPCurrent = gradP;
              }

              std::vector<value_type> outBlurredGradientData(inSize_ *
                  outGradientDataDim);*/
              //blurData_->blur_.compute(outGradData.data(), outGradientDataDim,
              blurData_->blur_.compute(grad.data(), valueSize_,
                  outOffset_, inOffset_, outSize_, inSize_,
                  dataGrad.data());
                  //outBlurredGradientData.data());

              /* // feature gradient -- in data
              // prepare the blur data
              const int inGradientDataDim = blurData_->featureSize_ * valueSize_;
              std::vector<value_type> inGradData(inGradientDataDim * inSize_);

              value_type* inGradP = inGradData.data();
              featureP = &feature[inOffset_*blurData_->featureSize_];
              const value_type* dataPCurrent = data.data();
              for (std::size_t i = 0; i < inSize_; ++i) {
                const value_type* dataP = nullptr;
                for (std::size_t d = 0; d < blurData_->featureSize_; ++d) {
                  dataP = dataPCurrent;
                  for (std::size_t e = 0; e < valueSize_; ++e) {
                    // data + feature
                    (*inGradP++) = (*dataP++) * (*featureP);
                  }

                  ++featureP;
                }

                dataPCurrent = dataP;
              }

              std::vector<value_type> inBlurredGradientData(outSize_ * inGradientDataDim);
              blurData_->blur_.compute(inGradData.data(), inGradientDataDim, inOffset_, outOffset_,
                  inSize_, outSize_, inBlurredGradientData.data());

              value_type* f = &featureGrad[inOffset_ *
                blurData_->featureSize_];
              dataPCurrent = data.data();
              featureP =
                &feature[inOffset_*blurData_->featureSize_];
              const value_type* outBlurredPCurrent = outBlurredGradientData.data();
              value_type* dataGradP = dataGrad.data();
              for (std::size_t i = 0; i < inSize_; ++i) {
                const value_type* dataP = nullptr;
                const value_type* outBlurredP = outBlurredPCurrent;
                const value_type* outBlurredGradP = nullptr;
                for (std::size_t d = 0; d < blurData_->featureSize_; ++d) {
                  *f = 0;

                  dataP = dataPCurrent;
                  outBlurredGradP = outBlurredPCurrent + outGradOffset;
                  for (std::size_t e = 0; e < valueSize_; ++e) {
                    *f += (*dataP) * (*outBlurredP++);

                    *f -= (*dataP) * (*featureP) * (*outBlurredGradP++);
                    ++dataP;
                  }

                  ++f;
                  ++featureP;
                }

                outBlurredGradP = outBlurredPCurrent + outGradOffset;
                for (std::size_t e = 0; e < valueSize_; ++e) {
                  (*dataGradP++) = (*outBlurredGradP++);
                }

                outBlurredPCurrent = outBlurredGradP;
                dataPCurrent = dataP;
              }


              featureP = &feature[outOffset_*blurData_->featureSize_];
              gradPCurrent = grad.data();
              f = &featureGrad[outOffset_*blurData_->featureSize_];
              const value_type* blurredPCurrent = blurred_.data();
              const value_type* inBlurredP = inBlurredGradientData.data();
              for (std::size_t i = 0; i < outSize_; ++i) {
                const value_type* gradP = nullptr;
                const value_type* blurredP = nullptr;
                for (std::size_t d = 0; d < blurData_->featureSize_; ++d) {
                  gradP = gradPCurrent;
                  blurredP = blurredPCurrent;

                  for (std::size_t e = 0; e < valueSize_; ++e) {

                    *f += (*gradP) * (*inBlurredP++);

                    *f -= (*featureP) * (*gradP) * (*blurredP++);
                    ++gradP;
                  }

                  ++f;
                  ++featureP;
                }

                blurredPCurrent = blurredP;
                gradPCurrent = gradP;
              }*/

              //blurData_->featureOp_->reverse(featureGrad);
              dataOp_->reverse(dataGrad);
            }

            const std::shared_ptr<BlurData> blurData_;
            const std::shared_ptr<ad::Op<value_type, 1>> dataOp_;

            vector_type blurred_;

            const std::size_t valueSize_;
            const std::size_t inOffset_;
            const std::size_t outOffset_;
            const std::size_t inSize_;
            const std::size_t outSize_;

            friend class DiffBlur;
        };

        std::shared_ptr<BlurData> blurData_;
    };

    template <typename TValue, typename TBlur>
    class SymmetricNormDiffBlur {
      public:
        typedef TValue value_type;
        typedef TBlur blur_type;

      private:
        typedef std::shared_ptr<ad::Op<value_type, 1>> vectorptr_type;

      public:
        void init(const std::shared_ptr<ad::Op<value_type, 1>>& featureOp,
            const std::size_t featureSize) {
          blur_.init(featureOp, featureSize);

          // calculate the norm
          const std::size_t dataCount = featureOp->value().size() / featureSize;

          typedef ad::Scalar<value_type, 1> scalar_type;
          typedef typename scalar_type::dim_type dim_type;
          std::shared_ptr<scalar_type> const1Op =
            std::make_shared<scalar_type>(
                (dim_type {{dataCount}}));

          value_type* const1 = const1Op->value().data();
          for (std::size_t i=0; i<dataCount; ++i)
            const1[i] = 1;

          const1Blur_ = blur_.compute(const1Op, 1, 0);
          const auto& blurred = const1Blur_->value();

          norm_ = std::make_shared<std::vector<value_type>>(dataCount);

          auto& norm = *norm_;

          for (std::size_t i = 0; i < dataCount; ++i) {
            norm[i] = 1.0 / std::sqrt(blurred[i] + 1e-20);
          }
        }

        std::shared_ptr<ad::Op<value_type, 1>>
        compute(const std::shared_ptr<ad::Op<value_type, 1>>& in, int
            valueSize, int inOffset=0, int outOffset=0, int outSize = -1)
        const {
          vectorptr_type normedIn =
            std::make_shared<SymmetricNormOp>(const1Blur_, norm_, in,
                valueSize, inOffset);

          vectorptr_type blurred = blur_.compute(normedIn, valueSize, inOffset,
              outOffset, outSize);

          return std::make_shared<SymmetricNormOp>(const1Blur_, norm_, blurred,
              valueSize, outOffset);
        }

      private:
        class SymmetricNormOp : public ad::RefOp<TValue, 1> {
          public:
            typedef TValue value_type;
            typedef typename ad::RefOp<value_type, 1>::vector_type vector_type;
            typedef typename ad::RefOp<value_type, 1>::dim_type dim_type;
            typedef typename std::shared_ptr<ad::Op<value_type, 1>> vectorptr_type;

            SymmetricNormOp(const vectorptr_type& const1Blur, const
                std::shared_ptr<const std::vector<value_type>>& normPtr, const
                vectorptr_type& inOp, int valueSize, int offset)
              : value_(dim_type {{inOp->value().size()}}),
                const1Blur_(const1Blur), normPtr_(normPtr), inOp_(inOp),
                valueSize_(valueSize), offset_(offset)
            {
              const1Blur_->reference();
              inOp_->reference();

              const value_type* in = inOp_->value().data();
              const auto& norm = *normPtr_;
              const int inSize = value_.size();
              for (int i = 0; i < inSize; ++i) {
                for (int e = 0; e < valueSize_; ++e) {
                  value_[i*valueSize_ + e] = in[i*valueSize_ + e] * norm[i +
                    offset_];
                }
              }
            }

            virtual
              const vector_type& value() const {
                return value_;
              }

          private:
            virtual
            void reverseImpl(const vector_type& gradIn) {
              const value_type* in = inOp_->value().data();
              const int inSize = value_.size();

              vector_type inGradOut(dim_type {{static_cast<std::size_t>(inSize)}});
              vector_type const1BlurGradOut(dim_type {{normPtr_->size()}});
              const auto& norm = *normPtr_;

              for (int i = 0; i < inSize; ++i) {
                for (int e = 0; e < valueSize_; ++e) {
                  inGradOut[i*valueSize_ + e] += gradIn[i*valueSize_ + e] *
                    norm[i + offset_];
                  const1BlurGradOut[i + offset_] -= 0.5 * gradIn[i*valueSize_ + e] *
                    in[i*valueSize_ + e] * std::pow(norm[i + offset_], 3);
                }
              }

              const1Blur_->reverse(const1BlurGradOut);
              inOp_->reverse(inGradOut);
            }

            vector_type value_;
            vectorptr_type const1Blur_;
            std::shared_ptr<const std::vector<value_type>> normPtr_;
            vectorptr_type inOp_;
            const int valueSize_;
            const int offset_;
        };


        DiffBlur<value_type, blur_type> blur_;
        std::shared_ptr<ad::Op<value_type, 1>> const1Blur_;
        std::shared_ptr<std::vector<value_type>> norm_;
    };
  }
}

#endif /* _DIFF_BLUR_798D92894727_ */
