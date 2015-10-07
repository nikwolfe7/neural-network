package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class LinearOutputFeedForwardDNNFactory extends FeedForwardDNNFactory {

  public LinearOutputFeedForwardDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
    super(example, hiddenLayerDimenions);
  }

  @Override
  protected NetworkElementAbstractFactory getNetworkElementFactory() {
    return new LinearOutputAbstractFactoryImpl();
  }

}
