package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class SigmoidNetworkFFDNNFactory extends FeedForwardDNNAbstractFactory {

  public SigmoidNetworkFFDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
    super(example, hiddenLayerDimenions);
  }

  @Override
  protected NetworkElementAbstractFactory getNetworkElementFactory() {
    return new SigmoidNetworkFactory();
  }

}
