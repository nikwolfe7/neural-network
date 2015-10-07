package mlsp.cs.cmu.edu.structure;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.elements.Edge;
import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.NetworkElement;
import mlsp.cs.cmu.edu.elements.Neuron;
import mlsp.cs.cmu.edu.elements.Output;
import mlsp.cs.cmu.edu.util.CostFunction;
import training.DataInstance;

/**
 * @author Nikolas Wolfe
 */
public class NeuralNetwork {
	
	private Input[] inputLayer;
	private Output[] outputLayer;
	private List<Layer> layers;
	/* These are the indices of the weight matrices */
	private List<Integer> weightMatrixLayers;
	/* These are the indices of the neuron layers */
	private List<Integer> hiddenNeuronLayers;
	/**
	 * We will infer that the first and last layers
	 * are the input and output layers, respectively
	 * 
	 * @param layers
	 */
	public NeuralNetwork(List<Layer> layers) {
		this.layers = layers;
		this.weightMatrixLayers = new ArrayList<>();
		this.hiddenNeuronLayers = new ArrayList<>();
		Layer input = layers.get(0);
		Layer output = layers.get(layers.size() - 1);
		this.inputLayer = new Input[input.size()];
		this.outputLayer = new Output[output.size()];
		for (int i = 0; i < input.size(); i++) {
			if (input.getElements()[i] instanceof Input)
				inputLayer[i] = (Input) input.getElements()[i];
		}
		for (int i = 0; i < output.size(); i++) {
			if (output.getElements()[i] instanceof Output)
				outputLayer[i] = (Output) output.getElements()[i];
		}
		for (int i = 1; i < layers.size() - 1; i++) {
			for (NetworkElement e : layers.get(i).getElements()) {
				if (e instanceof Edge) {
					weightMatrixLayers.add(i);
					break;
				} else if (e instanceof Neuron) {
					hiddenNeuronLayers.add(i);
					break;
				}
			}
		}
	}

	/**************************************************
	 * TESTING:
	 * 
	 * returns the outputs of the network
	 * 
	 * @param inputVector
	 * @return
	 */
	public double[] getPrediction(double[] inputVector) {
		for(int i = 0; i < inputVector.length; i++)
			inputLayer[i].setInputValue(inputVector[i]);
		for(int i = 0; i < layers.size(); i++) 
			layers.get(i).forward();
		return layers.get(layers.size()-1).getOutput();
	}
	
	/**************************************************
	 * Getters
	 */
	public double[] getHiddenLayerOutputs(int layer) {
		return layers.get(layer).getOutput();
	}
	
	public double[] getHiddenLayerDerivatives(int layer) {
		return layers.get(layer).getOutput();
	}
	
	public double[] getHiddenLayerErrorTerms(int layer) {
		return layers.get(layer).getErrorTerm();
	}
	
	public Layer getLayer(int layer) {
		return layers.get(layer);
	}
	
	public NetworkElement[] getLayerElements(int layer) {
		return layers.get(layer).getElements();
	}
	
	public int[] getWeightMatrixIndices() {
		int[] weights = new int[weightMatrixLayers.size()];
		for(int i = 0; i < weights.length; i++)
			weights[i] = weightMatrixLayers.get(i);
		return weights;
	}
	
	public int[] getHiddenLayerIndices() {
		int[] neurons = new int[hiddenNeuronLayers.size()];
		for(int i = 0; i < neurons.length; i++)
			neurons[i] = hiddenNeuronLayers.get(i);
		return neurons;
	}

	/***************************************************
	 * TRAINING: 
	 * 
	 * Returns the error for the instance
	 * 
	 * @param instance
	 * @return
	 */
	public double trainOnInstance(DataInstance instance) {
		double[] input = instance.getInputVector();
		double[] truth = instance.getOutputTruthValue();
		// forward propagate
		double[] prediction = getPrediction(input);
		// set output
		for(int i = 0; i < outputLayer.length; i++) 
			outputLayer[i].setTruthValue(truth[i]);
		// backward propagate
		for(int i = layers.size()-1; i >= 0; i--) 
			layers.get(i).backward();
		// sum error over outputs and return
		return CostFunction.meanSqError(prediction, truth);
	}

}
