package mlsp.cs.cmu.edu.factory;

import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.elements.Input;
import mlsp.cs.cmu.edu.elements.NetworkElement;
import mlsp.cs.cmu.edu.elements.Output;
import mlsp.cs.cmu.edu.structure.Layer;
import training.DataInstance;

/**
 * 
 * @author Nikolas Wolfe
 */
public class NeuralNetwork {
	
	private Input[] inputLayer;
	private Output[] outputLayer;
	private List<Layer> layers;
	/**
	 * We will infer that the first and last layers
	 * are the input and output layers, respectively
	 * 
	 * @param layers
	 */
	public NeuralNetwork(List<Layer> layers) {
		this.layers = layers;
		Layer input = layers.get(0);
		Layer output = layers.get(layers.size()-1);
		this.inputLayer = new Input[input.size()];
		this.outputLayer = new Output[output.size()];
		for(int i = 0; i < input.size(); i++) {
			inputLayer[i] = (Input) input.getElements()[i];
		}
		for(int i = 0; i < output.size(); i++) {
			outputLayer[i] = (Output) output.getElements()[i];
		}
	}

	public double[] getPrediction(double[] inputVector) {
		return inputVector;

	}

	/**
	 * Returns the error for the instance
	 * 
	 * @param instance
	 * @return
	 */
	public double trainOnInstance(DataInstance instance) {
		return 0;

	}

}
