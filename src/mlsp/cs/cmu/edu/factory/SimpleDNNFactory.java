package mlsp.cs.cmu.edu.factory;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import mlsp.cs.cmu.edu.elements.Bias;
import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.NetworkElement;
import mlsp.cs.cmu.edu.elements.Neuron;
import mlsp.cs.cmu.edu.elements.Output;
import mlsp.cs.cmu.edu.elements.Output;
import mlsp.cs.cmu.edu.structure.HiddenLayer;
import mlsp.cs.cmu.edu.structure.InputLayer;
import mlsp.cs.cmu.edu.structure.NetworkElementLayer;
import mlsp.cs.cmu.edu.structure.OutputLayer;

public class SimpleDNNFactory implements DNNFactory {
	
	private NetworkElementAbstractFactory f = new SimpleNetworkElementAbstractFactory();
	private NeuralNetwork network;

	public SimpleDNNFactory(int inputDimension, int outputDimension, int... hiddenLayerDimensions) {
		/**
		 *  Create input layer
		 */
		Queue<NetworkElementLayer> layers = new LinkedList<>();
		Input[] inputs = new Input[inputDimension];
		for (int i = 0; i < inputs.length; i++)
			inputs[i] = f.getInputElement();
		InputLayer inputLayer = new InputLayer(inputs);
		layers.add(inputLayer);
		/**
		 *  Create hidden layers
		 */
		for (int i = 0; i < hiddenLayerDimensions.length; i++) {
			int dim = hiddenLayerDimensions[i];
			Neuron[] hiddenLayerNeurons = new Neuron[dim];
			for(int j = 0; j < dim; j++) 
				hiddenLayerNeurons[j] = f.getNeuronElement();
			HiddenLayer hiddenLayer = new HiddenLayer(hiddenLayerNeurons);
			layers.add(hiddenLayer);
		}
		/**
		 *  Create output layers
		 */
		Output[] outputs = new Output[outputDimension];
		for (int i = 0; i < outputs.length; i++)
			outputs[i] = f.getOutputElement();
		OutputLayer outputLayer = new OutputLayer(outputs);
		layers.add(outputLayer);
		/**
		 *  Attach the layers / bias nodes
		 */
		List<NetworkElementLayer> connectedLayers = new LinkedList<>();
		NetworkElementLayer current = layers.poll(); // input layer
		Bias bias = f.getBiasElement();
		while(!layers.isEmpty()) {
			NetworkElementLayer next = layers.poll();
			Edge[] weights = current.connectLayer(next);
		}
		/**
		 *  Attach a bias
		 */

	}

	@Override
	public NeuralNetwork getInitializedNeuralNetwork() {
		// TODO Auto-generated method stub
		return null;
	}

}
