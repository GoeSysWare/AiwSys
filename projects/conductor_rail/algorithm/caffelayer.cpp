
#ifndef CAFFE_REG_H
#define CAFFE_REG_H

//layer

#include<caffe/layers/conv_layer.hpp> 
#include<caffe/layers/pooling_layer.hpp>
#include<caffe/layers/lrn_layer.hpp>
#include<caffe/layers/relu_layer.hpp>
#include<caffe/layers/sigmoid_layer.hpp>
#include<caffe/layers/softmax_layer.hpp>
#include<caffe/layers/tanh_layer.hpp>
#include<caffe/layers/python_layer.hpp>
#include<caffe/layers/absval_layer.hpp>
#include<caffe/layers/accuracy_layer.hpp>
#include<caffe/layers/argmax_layer.hpp>
#include<caffe/layers/batch_norm_layer.hpp>
#include<caffe/layers/batch_reindex_layer.hpp>
#include<caffe/layers/bias_layer.hpp>
#include<caffe/layers/bnll_layer.hpp>
#include<caffe/layers/concat_layer.hpp>
#include<caffe/layers/contrastive_loss_layer.hpp>
#include<caffe/layers/crop_layer.hpp>
#include<caffe/layers/data_layer.hpp>
#include<caffe/layers/deconv_layer.hpp>
#include<caffe/layers/dropout_layer.hpp>
#include<caffe/layers/dummy_data_layer.hpp>
#include<caffe/layers/eltwise_layer.hpp>
#include<caffe/layers/elu_layer.hpp>
#include<caffe/layers/embed_layer.hpp>
#include<caffe/layers/euclidean_loss_layer.hpp>
#include<caffe/layers/exp_layer.hpp>
#include<caffe/layers/filter_layer.hpp>
#include<caffe/layers/flatten_layer.hpp>
#include<caffe/layers/hdf5_data_layer.hpp>
#include<caffe/layers/hdf5_output_layer.hpp>
#include<caffe/layers/hinge_loss_layer.hpp>
#include<caffe/layers/im2col_layer.hpp>
#include<caffe/layers/image_data_layer.hpp>
#include<caffe/layers/infogain_loss_layer.hpp>
#include<caffe/layers/inner_product_layer.hpp>
#include<caffe/layers/input_layer.hpp>
#include<caffe/layers/log_layer.hpp>
#include<caffe/layers/lstm_layer.hpp>
#include<caffe/layers/memory_data_layer.hpp>
#include<caffe/layers/multinomial_logistic_loss_layer.hpp>
#include<caffe/layers/mvn_layer.hpp>
#include<caffe/layers/parameter_layer.hpp>
#include<caffe/layers/power_layer.hpp>
#include<caffe/layers/prelu_layer.hpp>
#include<caffe/layers/reduction_layer.hpp>
#include<caffe/layers/reshape_layer.hpp>
#include<caffe/layers/rnn_layer.hpp>
#include<caffe/layers/scale_layer.hpp>
#include<caffe/layers/sigmoid_cross_entropy_loss_layer.hpp>
#include<caffe/layers/silence_layer.hpp>
#include<caffe/layers/slice_layer.hpp>
#include<caffe/layers/softmax_loss_layer.hpp>
#include<caffe/layers/split_layer.hpp>
#include<caffe/layers/spp_layer.hpp>
#include<caffe/layers/threshold_layer.hpp>
#include<caffe/layers/tile_layer.hpp>
#include<caffe/layers/window_data_layer.hpp>

//solver
#include<caffe/sgd_solvers.hpp>


namespace caffe
{
	// 59 layers
	extern INSTANTIATE_CLASS(ConvolutionLayer);
	extern INSTANTIATE_CLASS(PoolingLayer);
	extern INSTANTIATE_CLASS(LRNLayer);
	extern INSTANTIATE_CLASS(ReLULayer);
	extern INSTANTIATE_CLASS(SigmoidLayer);
	extern INSTANTIATE_CLASS(SoftmaxLayer);
	extern INSTANTIATE_CLASS(TanHLayer);
	extern INSTANTIATE_CLASS(PythonLayer);
	extern INSTANTIATE_CLASS(AbsValLayer);
	extern INSTANTIATE_CLASS(AccuracyLayer);
	extern INSTANTIATE_CLASS(ArgMaxLayer);
	extern INSTANTIATE_CLASS(BatchNormLayer);
	extern INSTANTIATE_CLASS(BatchReindexLayer);
	extern INSTANTIATE_CLASS(BiasLayer);
	extern INSTANTIATE_CLASS(BNLLLayer);
	extern INSTANTIATE_CLASS(ConcatLayer);
	extern INSTANTIATE_CLASS(ContrastiveLossLayer);
	extern INSTANTIATE_CLASS(CropLayer);
	extern INSTANTIATE_CLASS(DataLayer);
	extern INSTANTIATE_CLASS(DeconvolutionLayer);
	extern INSTANTIATE_CLASS(DropoutLayer);
	extern INSTANTIATE_CLASS(DummyDataLayer);
	extern INSTANTIATE_CLASS(EltwiseLayer);
	extern INSTANTIATE_CLASS(ELULayer);
	extern INSTANTIATE_CLASS(EmbedLayer);
	extern INSTANTIATE_CLASS(EuclideanLossLayer);
	extern INSTANTIATE_CLASS(ExpLayer);
	extern INSTANTIATE_CLASS(FilterLayer);
	extern INSTANTIATE_CLASS(FlattenLayer);
	extern INSTANTIATE_CLASS(HDF5DataLayer);
	extern INSTANTIATE_CLASS(HDF5OutputLayer);
	extern INSTANTIATE_CLASS(HingeLossLayer);
	extern INSTANTIATE_CLASS(Im2colLayer);
	extern INSTANTIATE_CLASS(ImageDataLayer);
	extern INSTANTIATE_CLASS(InfogainLossLayer);
	extern INSTANTIATE_CLASS(InnerProductLayer);
	extern INSTANTIATE_CLASS(InputLayer);
	extern INSTANTIATE_CLASS(LogLayer);
	extern INSTANTIATE_CLASS(LSTMLayer);
	extern INSTANTIATE_CLASS(LSTMUnitLayer);
	extern INSTANTIATE_CLASS(MemoryDataLayer);
	extern INSTANTIATE_CLASS(MultinomialLogisticLossLayer);
	extern INSTANTIATE_CLASS(MVNLayer);
	extern INSTANTIATE_CLASS(ParameterLayer);
	extern INSTANTIATE_CLASS(PowerLayer);
	extern INSTANTIATE_CLASS(PReLULayer);
	extern INSTANTIATE_CLASS(ReductionLayer);
	extern INSTANTIATE_CLASS(ReshapeLayer);
	extern INSTANTIATE_CLASS(RNNLayer);
	extern INSTANTIATE_CLASS(ScaleLayer);
	extern INSTANTIATE_CLASS(SigmoidCrossEntropyLossLayer);
	extern INSTANTIATE_CLASS(SilenceLayer);
	extern INSTANTIATE_CLASS(SliceLayer);
	extern INSTANTIATE_CLASS(SoftmaxWithLossLayer);
	extern INSTANTIATE_CLASS(SplitLayer);
	extern INSTANTIATE_CLASS(SPPLayer);
	extern INSTANTIATE_CLASS(ThresholdLayer);
	extern INSTANTIATE_CLASS(TileLayer);
	extern INSTANTIATE_CLASS(WindowDataLayer);

	// 6 sovlers
	extern INSTANTIATE_CLASS(AdaDeltaSolver);
	extern INSTANTIATE_CLASS(AdaGradSolver);
	extern INSTANTIATE_CLASS(AdamSolver);
	extern INSTANTIATE_CLASS(NesterovSolver);
	extern INSTANTIATE_CLASS(RMSPropSolver);
	extern INSTANTIATE_CLASS(SGDSolver);
}
#endif