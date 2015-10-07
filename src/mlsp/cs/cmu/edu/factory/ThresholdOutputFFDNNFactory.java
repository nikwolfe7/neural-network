package mlsp.cs.cmu.edu.factory;

import training.DataInstance;

public class ThresholdOutputFFDNNFactory extends FeedForwardDNNFactory {

  public ThresholdOutputFFDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
    super(example, hiddenLayerDimenions);
  }

  @Override
  protected NetworkElementAbstractFactory getNetworkElementFactory() {
    return new ThresholdOutputAbstractFactoryImpl();
  }

}
