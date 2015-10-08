package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class ThresholdOutputFFDNNFactory extends FeedForwardDNNAbstractFactory {

  public ThresholdOutputFFDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
    super(example, hiddenLayerDimenions);
  }

  @Override
  protected NetworkElementAbstractFactory getNetworkElementFactory() {
    return new ThresholdOutputFactory();
  }

}
