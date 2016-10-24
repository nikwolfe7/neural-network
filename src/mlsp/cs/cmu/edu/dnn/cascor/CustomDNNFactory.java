package mlsp.cs.cmu.edu.dnn.cascor;

import java.util.List;

import mlsp.cs.cmu.edu.dnn.elements.AdaGradEdge;
import mlsp.cs.cmu.edu.dnn.elements.Bias;
import mlsp.cs.cmu.edu.dnn.elements.CrossEntropyOutput;
import mlsp.cs.cmu.edu.dnn.elements.CrossEntropyTanhOutput;
import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.Input;
import mlsp.cs.cmu.edu.dnn.elements.MomentumEdge;
import mlsp.cs.cmu.edu.dnn.elements.Neuron;
import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.elements.SimpleEdge;
import mlsp.cs.cmu.edu.dnn.elements.TanhNeuron;
import mlsp.cs.cmu.edu.dnn.factory.FeedForwardDNNAbstractFactory;
import mlsp.cs.cmu.edu.dnn.factory.NetworkElementAbstractFactory;
import mlsp.cs.cmu.edu.dnn.structure.CrossEntropyNeuralNetwork;
import mlsp.cs.cmu.edu.dnn.structure.Layer;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class CustomDNNFactory extends FeedForwardDNNAbstractFactory {

	public CustomDNNFactory(DataInstance example, int[] hiddenLayerDimenions) {
		super(example, hiddenLayerDimenions);
	}

	@Override
	protected NeuralNetwork getNewNeuralNetwork(List<Layer> layers) {
		return new NeuralNetwork(layers);
	}

	@Override
	protected NetworkElementAbstractFactory getNetworkElementFactory() {
		return new NetworkElementAbstractFactory() {

			@Override
			public Output getNewOutput() {
				return new Output();
			}

			@Override
			public Neuron getNewNeuron() {
				return new Neuron();
			}

			@Override
			public Input getNewInput() {
				return new Input();
			}

			@Override
			public Edge getNewEdge() {
				return new MomentumEdge(-1, 1, 0.0005, 0.95);
//				return new SimpleEdge(-1, 1, 0.001);
			}

			@Override
			public Bias getNewBias() {
				return new Bias();
			}
		};
	}
}
