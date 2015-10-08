package mlsp.cs.cmu.edu.dnn.factory;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.elements.*;
import mlsp.cs.cmu.edu.dnn.structure.*;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public abstract class FeedForwardDNNAbstractFactory implements DNNFactory {

	private NeuralNetwork network;
	private NetworkElementAbstractFactory factory;

	public FeedForwardDNNAbstractFactory(DataInstance example, int... hiddenLayerDimenions) {
		this.factory = getNetworkElementFactory();
		int inputDimension = example.getInputDimension();
		int outputDimension = example.getOutputDimension();
		List<Layer> layers = new ArrayList<>();

		/* Input layer */
		Input[] inputs = new Input[inputDimension];
		for (int i = 0; i < inputs.length; i++) {
			inputs[i] = factory.getNewInput();
		}
		Layer inputLayer = new NetworkElementLayer(inputs);
		layers.add(inputLayer);

		/* Hidden Layers */
		for (int i = 0; i < hiddenLayerDimenions.length; i++) {
			/* previous layer */
			Layer prev = layers.get(layers.size() - 1);
			int dim = hiddenLayerDimenions[i];
			Neuron[] hl = new Neuron[dim];
			for (int j = 0; j < hl.length; j++) {
				hl[j] = factory.getNewNeuron();
			}
			Layer hiddenLayer = new NetworkElementLayer(hl);
			/* Add the Bias */
			prev.addNetworkElements(factory.getNewBias());
			/* Connect the layers */
			Layer weightMatrix = connect(prev, hiddenLayer);
			/* Load them into the network stack */
			layers.add(weightMatrix);
			layers.add(hiddenLayer);
		}

		/* Output layer */
		Output[] outputs = new Output[outputDimension];
		for (int i = 0; i < outputs.length; i++) {
			outputs[i] = factory.getNewOutput();
		}
		Layer outputLayer = new NetworkElementLayer(outputs);
		Layer prev = layers.get(layers.size() - 1);
		/* output layer bias... */
		prev.addNetworkElements(factory.getNewBias());
		Layer weightMatrix = connect(prev, outputLayer);
		layers.add(weightMatrix);
		layers.add(outputLayer);
		this.network = new NeuralNetwork(layers);
	}

  private Layer connect(Layer prev, Layer hiddenLayer) {
		int rows, cols;
		rows = hiddenLayer.size();
		cols = prev.size();
		Edge[] weightMatrix = new Edge[rows * cols];
		for (int row = 0; row < rows; row++) {
			for (int col = 0; col < cols; col++) {
				Neuron out = (Neuron) hiddenLayer.getElements()[row];
				Neuron in = (Neuron) prev.getElements()[col];
				Edge w = factory.getNewEdge();
				attachElements(in, w, out);
				weightMatrix[row * cols + col] = w;
			}
		}
		return new NetworkElementLayer(weightMatrix);
	}

	private void attachElements(Neuron in, Edge w, Neuron out) {
		w.setIncomingElement(in);
		w.setOutgoingElement(out);
		in.addOutgoingElement(w);
		out.addIncomingElement(w);
	}

	@Override
	public NeuralNetwork getInitializedNeuralNetwork() {
		return network;
	}
	
	protected abstract NetworkElementAbstractFactory getNetworkElementFactory();

}
