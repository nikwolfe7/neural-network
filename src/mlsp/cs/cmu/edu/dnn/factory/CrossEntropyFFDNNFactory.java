package mlsp.cs.cmu.edu.dnn.factory;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.structure.CrossEntropyNeuralNetwork;
import mlsp.cs.cmu.edu.dnn.structure.Layer;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class CrossEntropyFFDNNFactory extends FeedForwardDNNAbstractFactory {

  public CrossEntropyFFDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
    super(example, hiddenLayerDimenions);
  }

  @Override
  protected NeuralNetwork getNewNeuralNetwork(List<Layer> layers) {
    return new CrossEntropyNeuralNetwork(layers);
  }

  @Override
  protected NetworkElementAbstractFactory getNetworkElementFactory() {
    return new CrossEntropyNetworkElementFactory();
  }

}
