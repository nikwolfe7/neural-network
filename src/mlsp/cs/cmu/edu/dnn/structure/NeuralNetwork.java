package mlsp.cs.cmu.edu.dnn.structure;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.Input;
import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;
import mlsp.cs.cmu.edu.dnn.elements.Neuron;
import mlsp.cs.cmu.edu.dnn.elements.Output;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.util.CostFunction;
import mlsp.cs.cmu.edu.dnn.util.OutputAdapter;

/**
 * @author Nikolas Wolfe
 */
public class NeuralNetwork implements Serializable {

	private static final long serialVersionUID = 5761924668947132220L;

	private Input[] inputLayer;
	private Output[] outputLayer;
	private int inputDimension;
	private int outputDimension;
	private List<Layer> layers;

	/* These are the indices of the weight matrices */
	private List<Integer> weightMatrixLayers;
	/* These are the indices of the neuron layers */
	private List<Integer> hiddenNeuronLayers;

	/* This produces a "clean" prediction, i.e. smoothed or formatted */
	/**
	 * We will infer that the first and last layers are the input and output
	 * layers, respectively
	 * 
	 * @param layers
	 */
	public NeuralNetwork(List<Layer> layers) {
		this.layers = layers;
		this.weightMatrixLayers = new ArrayList<>();
		this.hiddenNeuronLayers = new ArrayList<>();
		Layer input = layers.get(0);
		Layer output = layers.get(layers.size() - 1);
		this.inputDimension = 0;
		this.outputDimension = 0;
		Input[] in = new Input[input.size()];
		Output[] out = new Output[output.size()];
		for (int i = 0; i < input.size(); i++) {
			if (input.getElements()[i] instanceof Input) {
				in[i] = (Input) input.getElements()[i];
				inputDimension++;
			}
		}
		for (int i = 0; i < output.size(); i++) {
			if (output.getElements()[i] instanceof Output) {
				out[i] = (Output) output.getElements()[i];
				outputDimension++;
			}
		}
		this.inputLayer = new Input[inputDimension];
		this.outputLayer = new Output[outputDimension];
		System.arraycopy(in, 0, inputLayer, 0, inputLayer.length);
		System.arraycopy(out, 0, outputLayer, 0, outputLayer.length);
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

	public int getInputDimension() {
		return inputDimension;
	}

	public int getOutputDimension() {
		return outputDimension;
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
		for (int i = 0; i < inputVector.length; i++)
			inputLayer[i].setInputValue(inputVector[i]);
		for (int i = 0; i < layers.size(); i++)
			layers.get(i).forward();
		return getOutputs();
	}

	/**************************************************
	 * FOR RUNNING: returns some smoothed output, i.e. text or a smoothed
	 * number, or something like that.
	 * 
	 * The recipient of this output knows how to cast it appropriately
	 * 
	 * @param inputVector
	 * @return
	 */
	public Object getSmoothedPrediction(double[] inputVector, OutputAdapter adapter) {
		double[] output = getPrediction(inputVector);
		return adapter.getSmoothedPrediction(output);
	}

	/**************************************************
	 * Getters
	 */
	public double[] getOutputs() {
		return getLastLayer().getOutput();
	}

	public double[] getLayerOutputs(int layer) {
		return layers.get(layer).getOutput();
	}

	public double[] getLayerDerivatives(int layer) {
		return layers.get(layer).derivative();
	}

	public double[] getLayerGradients(int layer) {
		return layers.get(layer).getGradient();
	}

	public Layer getLayer(int layer) {
		return layers.get(layer);
	}

	public Layer getFirstLayer() {
		return layers.get(0);
	}

	public Layer getLastLayer() {
		return layers.get(layers.size() - 1);
	}

	public NetworkElement[] getLayerElements(int layer) {
		return layers.get(layer).getElements();
	}

	public int[] getWeightMatrixIndices() {
		int[] weights = new int[weightMatrixLayers.size()];
		for (int i = 0; i < weights.length; i++)
			weights[i] = weightMatrixLayers.get(i);
		return weights;
	}

	public int[] getHiddenLayerIndices() {
		int[] neurons = new int[hiddenNeuronLayers.size()];
		for (int i = 0; i < neurons.length; i++)
			neurons[i] = hiddenNeuronLayers.get(i);
		return neurons;
	}

	/**************************************************
	 * Weight operations
	 */
	public void setGlobalLearningRate(double newRate) {
		for (int index : getWeightMatrixIndices()) {
			for (NetworkElement e : getLayerElements(index)) {
				Edge edge = (Edge) e;
				edge.setLearningRate(newRate);
			}
		}
	}

	public void reinitializeWeights(double low, double high) {
		for (int index : getWeightMatrixIndices()) {
			for (NetworkElement e : getLayerElements(index)) {
				Edge edge = (Edge) e;
				edge.reinitializeWeight(low, high);
			}
		}
	}

	public void setBatchUpdate(boolean b) {
		for (int index : getWeightMatrixIndices()) {
			for (NetworkElement e : getLayerElements(index)) {
				Edge edge = (Edge) e;
				edge.setBatchUpdate(b);
			}
		}
	}

	public void batchUpdate() {
		int[] weightIndices = getWeightMatrixIndices();
		for (int index : weightIndices) {
			for (NetworkElement e : getLayerElements(index)) {
				Edge edge = (Edge) e;
				edge.batchUpdate();
			}
		}
	}
	
	/***************************************************
	 * Layer Operations - allows an existing Layer in 
	 * the network to be replaced...  Assumes connections 
	 * have been already established. 
	 * 
	 * Use with caution.
	 * 
	 * No guarantees that anything will work if you don't
	 * know explicitly what you are doing. 
	 * 
	 * @param i
	 * @param modified
	 */
	public void modifyExistingLayer(Layer oldLayer, Layer newLayer) {
		layers.set(layers.indexOf(oldLayer), newLayer);
	}
	
	/**
	 * Extricates the elements in the list from the network, cleans up
	 * any artifacts that may be left over... 
	 * 
	 * @param elements
	 */
  public void removeElements(List<NetworkElement> elements) {
    /* Remove neurons in the hidden layers */
    for (int i : getHiddenLayerIndices()) {
      Layer layer = getLayer(i);
      for (NetworkElement e : elements)
        layer.removeNetworkElement(e);
    }
    /* Remove weights in the hidden layers */
    for (int i : getWeightMatrixIndices()) {
      Layer layer = getLayer(i);
      for (NetworkElement e : elements)
        layer.removeNetworkElement(e);
    }
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
		for (int i = 0; i < outputLayer.length; i++)
			outputLayer[i].setTruthValue(truth[i]);
		// backward propagate
		for (int i = layers.size() - 1; i >= 0; i--)
			layers.get(i).backward();
		// sum error over outputs and return
		return getErrorTerm(prediction, truth);
	}

	/* Override in a sublcass to do cross entropy */
	protected double getErrorTerm(double[] prediction, double[] truth) {
		return CostFunction.meanSqError(prediction, truth);
	}

}
