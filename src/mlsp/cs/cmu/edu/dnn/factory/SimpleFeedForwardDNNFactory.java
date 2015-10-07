package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class SimpleFeedForwardDNNFactory extends FeedForwardDNNFactory {

  public SimpleFeedForwardDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
    super(example, hiddenLayerDimenions);
  }

  @Override
  protected NetworkElementAbstractFactory getNetworkElementFactory() {
    return new SigmoidNetworkAbstractFactoryImpl();
  }

}
