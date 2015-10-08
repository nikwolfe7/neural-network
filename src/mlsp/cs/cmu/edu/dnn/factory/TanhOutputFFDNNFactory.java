package mlsp.cs.cmu.edu.dnn.factory;

import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class TanhOutputFFDNNFactory extends FeedForwardDNNAbstractFactory {

	public TanhOutputFFDNNFactory(DataInstance example, int... hiddenLayerDimenions) {
		super(example, hiddenLayerDimenions);
	}

	@Override
	protected NetworkElementAbstractFactory getNetworkElementFactory() {
		return new TanhOutputFactory();
	}

}
