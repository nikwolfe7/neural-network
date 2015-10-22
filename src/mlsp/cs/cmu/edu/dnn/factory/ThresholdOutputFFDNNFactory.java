package mlsp.cs.cmu.edu.dnn.factory;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.structure.Layer;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class ThresholdOutputFFDNNFactory extends FeedForwardDNNAbstractFactory {

  public ThresholdOutputFFDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
    super(example, hiddenLayerDimenions);
  }

  @Override
  protected NetworkElementAbstractFactory getNetworkElementFactory() {
    return new ThresholdOutputElementFactory();
  }

  @Override
  protected NeuralNetwork getNewNeuralNetwork(List<Layer> layers) {
    return new NeuralNetwork(layers);
  }

}
