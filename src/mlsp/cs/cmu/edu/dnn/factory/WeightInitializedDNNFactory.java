package mlsp.cs.cmu.edu.dnn.factory;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.training.MNISTSmallDataInstanceFactory;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;

public class WeightInitializedDNNFactory implements DNNFactory {

	NeuralNetwork dnn = null;
	
	/* Weight matrices are assumed to be in a format of LAYER_DIMENSION X INPUT_DIMENSION */
	public WeightInitializedDNNFactory(DataInstance example, String... matrixFiles) {
		if(matrixFiles.length < 1 || example == null) {
			System.out.println("No matrix files given or training instance is null... Exiting!");
			System.exit(0);
		}
		List<List<double[]>> weightMatrices = new ArrayList<>();
		for(String file : matrixFiles) {
			weightMatrices.add(getWeightMatrixFromFile(file));
		}
		// Number of hidden layers == matrixFiles.length - 1
		int numHiddenLayers = matrixFiles.length - 1;
		System.out.println("Num hidden layers: " + numHiddenLayers);
		// hidden layer dimensions
		int[] structure = new int[numHiddenLayers];
		for(int i = 0; i < numHiddenLayers; i++) {
			structure[i] = weightMatrices.get(i).size();
		}
		System.out.println("Hidden layer dimensions: " + DNNUtils.joinNumbers(structure, ", "));
		// Sanity checks
		int inputDimension = weightMatrices.get(0).get(0).length - 1; 
		System.out.println("Matrix input dimension: " + inputDimension + "\nDataInstance input dimension: " + example.getInputDimension());
		if(inputDimension != example.getInputDimension()) {
			System.out.println("Weight input dimensions do not match... Exiting!");
			System.exit(0);
		}
		int outputDimension = weightMatrices.get(numHiddenLayers).size();
		System.out.println("Matrix output dimension: " + outputDimension + "\nDataInstance input dimension: " + example.getOutputDimension());
		if(outputDimension != example.getOutputDimension()) {
			System.out.println("Weight input dimensions do not match... Exiting!");
			System.exit(0);
		}
		
	}
	

	
	private List<double[]> getWeightMatrixFromFile(String file) {
		List<double[]> data = new ArrayList<>();
		try {
			  System.out.println("Getting weights from file " + file);
		      Scanner scn = new Scanner(new File(file));
		      while(scn.hasNextLine()) {
		        String[] line = scn.nextLine().split("\\s+|,");
		        double[] arr = new double[line.length];
		        for(int i = 0; i < line.length; i++)
		          arr[i] = Double.parseDouble(line[i]);
		        data.add(arr);
		      }
		      scn.close();
		    } catch (FileNotFoundException e) {
		      e.printStackTrace();
		    }
		return data;
	}

	@Override
	public NeuralNetwork getInitializedNeuralNetwork() {
		return this.dnn;
	}
	
	public static void main(String[] args) {
		String[] matrixFiles = {"data" + DNNUtils.sep + "mat-100x401.csv", "data" + DNNUtils.sep + "mat-10x101.csv"};
		DataInstanceFactory dataInstanceFactory = new MNISTSmallDataInstanceFactory();
		List<DataInstance> training = dataInstanceFactory.getTrainingInstances();
		DNNFactory factory = new WeightInitializedDNNFactory(training.get(0), matrixFiles);
		
	}

}
