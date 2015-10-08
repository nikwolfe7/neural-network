package mlsp.cs.cmu.edu.dnn.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class DNNUtils {

	public static String printVector(double[] vec) {
		DecimalFormat f = new DecimalFormat("###.###");
		StringBuilder sb = new StringBuilder("[  ");
		for (Double d : vec)
			sb.append(f.format(d) + "  ");
		return sb.toString() + "]";
	}

	public static List<double[]> getInputsFromFile(String file) throws IOException {
		List<double[]> inputs = new ArrayList<double[]>();
		Scanner scn = new Scanner(new File(file));
		while (scn.hasNextLine()) {
			String line = scn.nextLine();
			String[] arr = line.split("\\,");
			double[] vector = new double[arr.length];
			for (int i = 0; i < arr.length; i++)
				vector[i] = Double.valueOf(arr[i]);
			inputs.add(vector);
		}
		scn.close();
		return inputs;
	}

	public static List<DataInstance> getTrainingInstances(String file, int inputDimension, int outputDimension)
			throws IOException {
		List<double[]> data = getInputsFromFile(file);
		List<DataInstance> instances = new ArrayList<DataInstance>();
		for (double[] instance : data)
			instances.add(new DataInstance(inputDimension, outputDimension, instance));
		return instances;
	}

	public static List<DataInstance> minMaxRangeNormalizeInputs(List<DataInstance> data, double low, double high) {
		List<double[]> inputs = new ArrayList<>();
		for(int i = 0; i < data.size(); i++)
			inputs.add(data.get(i).getInputVector());
		inputs = minMaxRangeNormalize(inputs, low, high);
		for(int i = 0; i < data.size(); i++) 
			data.get(i).replaceInputVector(inputs.get(i));
		return data;
	}
	
	public static List<DataInstance> minMaxRangeNormalizeOutputs(List<DataInstance> data, double low, double high) {
		List<double[]> outputs = new ArrayList<>();
		for(int i = 0; i < data.size(); i++)
			outputs.add(data.get(i).getOutputTruthValue());
		outputs = minMaxRangeNormalize(outputs, low, high);
		for(int i = 0; i < data.size(); i++) 
			data.get(i).replaceOutputTruthValue(outputs.get(i));
		return data;
	}

	public static List<DataInstance> zScoreNormalizeInputs(List<DataInstance> data) {
		List<double[]> inputs = new ArrayList<>();
		for(int i = 0; i < data.size(); i++)
			inputs.add(data.get(i).getInputVector());
		inputs = zScoreNormalize(inputs);
		for(int i = 0; i < data.size(); i++) 
			data.get(i).replaceInputVector(inputs.get(i));
		return data;
	}
	
	public static List<DataInstance> zScoreNormalizeOutputs(List<DataInstance> data) {
		List<double[]> outputs = new ArrayList<>();
		for(int i = 0; i < data.size(); i++)
			outputs.add(data.get(i).getOutputTruthValue());
		outputs = zScoreNormalize(outputs);
		for(int i = 0; i < data.size(); i++) 
			data.get(i).replaceOutputTruthValue(outputs.get(i));
		return data;
	}
	
	public static List<DataInstance> meanSubtractNormalizeInputs(List<DataInstance> data) {
		List<double[]> inputs = new ArrayList<>();
		for(int i = 0; i < data.size(); i++)
			inputs.add(data.get(i).getInputVector());
		inputs = zScoreNormalize(inputs);
		for(int i = 0; i < data.size(); i++) 
			data.get(i).replaceInputVector(inputs.get(i));
		return data;
	}
	
	public static List<DataInstance> meanSubtractNormalizeOutputs(List<DataInstance> data) {
		List<double[]> outputs = new ArrayList<>();
		for(int i = 0; i < data.size(); i++)
			outputs.add(data.get(i).getOutputTruthValue());
		outputs = meanSubtractNormalize(outputs);
		for(int i = 0; i < data.size(); i++) 
			data.get(i).replaceOutputTruthValue(outputs.get(i));
		return data;
	}
	
	public static List<double[]> minMaxRangeNormalize(List<double[]> data, double low, double high) {
		for (int j = 0; j < data.get(0).length; j++) {
			double min = data.get(0)[j];
			double max = min;
			/* calc range */
			for (int i = 0; i < data.size(); i++) {
				double[] x = data.get(i);
				min = Math.min(min, x[j]);
				max = Math.max(max, x[j]);
			}
			/* normalize on range */
			for (int i = 0; i < data.size(); i++) {
				double[] x = data.get(i);
				x[j] = (x[j] - min) / (max - min);
				x[j] = x[j] * (high - low) + low;
				data.set(i, x);
			}
		}
		return data;
	}
	
	public static List<double[]> zScoreNormalize(List<double[]> data) {
		for (int j = 0; j < data.get(0).length; j++) {
			double sum = 0;
			double N = data.size();
			/* calc avg */
			for (int i = 0; i < data.size(); i++) {
				double[] x = data.get(i);
				sum += x[j];
			}
			double mean = sum / N;
			/* calc variance */
			sum = 0;
			for (int i = 0; i < data.size(); i++) {
				double[] x = data.get(i);
				sum += Math.pow((x[j] - mean), 2);
			}
			double variance = sum / (N - 1);
			double stdev = Math.sqrt(variance);
			for (int i = 0; i < data.size(); i++) {
				double[] x = data.get(i);
				x[j] = (x[j] - mean) / stdev;
				data.set(i, x);
			}
		}
		return data;
	}
	
	public static List<double[]> meanSubtractNormalize(List<double[]> data) {
		for (int j = 0; j < data.get(0).length; j++) {
			double sum = 0;
			double N = data.size();
			/* calc avg */
			for (int i = 0; i < data.size(); i++) {
				double[] x = data.get(i);
				sum += x[j];
			}
			double mean = sum / N;
			for (int i = 0; i < data.size(); i++) {
				double[] x = data.get(i);
				x[j] = (x[j] - mean);
				data.set(i, x);
			}
		}
		return data;
	}

}
