package mlsp.cs.cmu.edu.factory;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.structure.Layer;

public class SimpleDNNFactory implements DNNFactory {

	private NeuralNetwork network;
	
	public SimpleDNNFactory(int inputDimension, int outputDimension, int... hiddenLayerDimenions) {
		List<Layer> layers = new ArrayList<>();
	}
	
	
	@Override
	public NeuralNetwork getInitializedNeuralNetwork() {
		return network;
	}
	
	

}
