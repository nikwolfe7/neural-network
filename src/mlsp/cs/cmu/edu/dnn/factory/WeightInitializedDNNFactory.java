package mlsp.cs.cmu.edu.dnn.factory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import mlsp.cs.cmu.edu.dnn.elements.Edge;
import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTSmallDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;

public abstract class WeightInitializedDNNFactory implements DNNFactory {

	NeuralNetwork dnn = null;
	boolean biasComesFirst = false;

	/*
	 * Weight matrices are assumed to be in a format of:
	 * 
	 * LAYER_DIMENSION X INPUT_DIMENSION
	 */
	public WeightInitializedDNNFactory(DataInstance example, boolean biasFirst, String... matrixFiles) {
		this.biasComesFirst = biasFirst;
		if (matrixFiles.length < 1 || example == null) {
			System.out.println("No matrix files given or training instance is null... Exiting!");
			System.exit(0);
		}
		List<List<double[]>> weightMatrices = new ArrayList<>();
		for (String file : matrixFiles) {
			weightMatrices.add(getWeightMatrixFromFile(file));
		}
		// Number of hidden layers == matrixFiles.length - 1
		int numHiddenLayers = matrixFiles.length - 1;
		System.out.println("Num hidden layers: " + numHiddenLayers);
		// hidden layer dimensions
		int[] structure = new int[numHiddenLayers];
		for (int i = 0; i < numHiddenLayers; i++) {
			structure[i] = weightMatrices.get(i).size();
		}
		System.out.println("Hidden layer dimensions: " + DNNUtils.joinNumbers(structure, ", "));
		// Sanity checks
		int inputDimension = weightMatrices.get(0).get(0).length - 1;
		System.out.println("Matrix input dimension: " + inputDimension + "\nDataInstance input dimension: "
				+ example.getInputDimension());
		if (inputDimension != example.getInputDimension()) {
			System.out.println("Weight input dimensions do not match... Exiting!");
			System.exit(0);
		}
		int outputDimension = weightMatrices.get(numHiddenLayers).size();
		System.out.println("Matrix output dimension: " + outputDimension + "\nDataInstance input dimension: "
				+ example.getOutputDimension());
		if (outputDimension != example.getOutputDimension()) {
			System.out.println("Weight input dimensions do not match... Exiting!");
			System.exit(0);
		}
		// We have the dimensions, so init a neural network...
		FeedForwardDNNAbstractFactory dnnFactory = getFFDNNFactory(example, structure);
		this.dnn = dnnFactory.getInitializedNeuralNetwork();
		doInitializeWeightParameters(weightMatrices);
	}

	private List<double[]> getWeightMatrixFromFile(String file) {
		List<double[]> data = new ArrayList<>();
		try {
			System.out.println("Getting weights from file " + file);
			Scanner scn = new Scanner(new File(file));
			while (scn.hasNextLine()) {
				String[] line = scn.nextLine().split("\\s+|,");
				double[] arr = new double[line.length];
				for (int i = 0; i < line.length; i++)
					arr[i] = Double.parseDouble(line[i]);
				data.add(arr);
			}
			scn.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		return data;
	}

	private void doInitializeWeightParameters(List<List<double[]>> weightMatrices) {
		/* Get indices of weight matrices in the DNN */
		int[] weightMatrixIndices = dnn.getWeightMatrixIndices();
		for (int i = 0; i < weightMatrices.size(); ++i) {
			System.out.println("Setting weight matrix " + (i + 1) + "...");
			/* Get the i-th weight matrix from the input weight matrices */
			List<double[]> inputWeightMatrix = weightMatrices.get(i);
			/* Get the 1D weight matrix of the DNN */
			NetworkElement[] dnnWeightMatrix = dnn.getLayerElements(weightMatrixIndices[i]);
			for (int row = 0; row < inputWeightMatrix.size(); row++) {
				// System.out.println("Neuron index: " + row);
				/* Get input weight vector row for neuron */
				double[] neuronWeightVector = inputWeightMatrix.get(row);
				/*
				 * Used because we index 1D weight matrix as 2D with row + col
				 */
				int nCols = neuronWeightVector.length;
				/* Handle for bias occuring first in the vector */
				if (biasComesFirst) {
					/*
					 * set bias to last element of neuron vector in weight
					 * matrix
					 */
					int index = row * nCols + nCols - 1;
					double bias = neuronWeightVector[0];
					Edge e = (Edge) dnnWeightMatrix[index];
					e.setWeight(bias);
				}
				/* Iterate through weight vector for single neuron */
				for (int col = 0; col < nCols; col++) {
					int edgeIndex = row * nCols + col;
					Edge e = (Edge) dnnWeightMatrix[edgeIndex];
					int weightIndex = (biasComesFirst) ? col + 1 : col;
					if (weightIndex >= nCols)
						break;
					double w = neuronWeightVector[weightIndex];
					e.setWeight(w);
				}
			}
		}
		System.out.println("Done initializing weights!");
	}

	@Override
	public NeuralNetwork getInitializedNeuralNetwork() {
		return this.dnn;
	}

	public abstract FeedForwardDNNAbstractFactory getFFDNNFactory(DataInstance example, int... hiddenLayerDimenions);

}
