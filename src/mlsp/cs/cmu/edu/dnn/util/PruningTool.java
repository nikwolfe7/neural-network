package mlsp.cs.cmu.edu.dnn.util;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedList;
import java.util.List;
import java.util.Map;
import java.util.Random;

import mlsp.cs.cmu.edu.dnn.elements.GainSwitchNeuron;
import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;
import mlsp.cs.cmu.edu.dnn.elements.SwitchEdge;
import mlsp.cs.cmu.edu.dnn.elements.Switchable;
import mlsp.cs.cmu.edu.dnn.structure.GainSwitchLayer;
import mlsp.cs.cmu.edu.dnn.structure.Layer;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class PruningTool {

	public static OutputAdapter adapter = new BinaryThresholdOutput();
	public static boolean printOut = true;
	public static boolean batchUpdate = false;
	public static String sep = System.getProperty("file.separator");
	public static String data = "." + sep + "data" + sep;

	public static NeuralNetwork doPruning(String dnnFile, boolean newFile, NeuralNetwork net, List<DataInstance> training, List<DataInstance> testing, double percentReduce) throws IOException {
		DNNTrainingModule trainingModule = null;
		if (newFile) {
			/* Get training module and run through the validation set once */
			System.out.println("Modifying network for 2nd derivative backprop...");
			modifyLayers(net);
			System.out.println("Initializing training module...");
			trainingModule = initTrainingModule(net, training, testing);
			System.out.println("Running through validation set to compute first and second gradients...");
			trainingModule.doTrainNetworkUntilConvergence();
		}
		List<GainSwitchNeuron> neuronsToSort = getGainSwitchNeurons(net);
		/*
		 * Get the ground truth for the change in error resulting from turning
		 * the neurons on and off
		 */
		if (newFile) {
			System.out.println("Computing ground truth error...");
			getGroundTruthError(neuronsToSort, net, training);
			trainingModule.saveNetworkToFile("mod." + dnnFile);
		}
		/* Sort the list by the different metrics */
		System.out.println("Sorting by ground truth, gain, and second gain...");
		int[] groundTruthRankings = sortNeurons("gt",neuronsToSort);
		double[] groundTruthErrorRank = getSortByValues("gt",neuronsToSort);
		
		int[] gainSumRankings = sortNeurons("g1",neuronsToSort);
		double[] gainSumErrorRank = getSortByValues("g1",neuronsToSort);
		
		int[] secondGainSumRankings = sortNeurons("g2", neuronsToSort);
		double[] secondGainSumErrorRank = getSortByValues("g2",neuronsToSort);
		
		double[] dropOffForGT = singlePassAlgorithm(percentReduce, net, training, "gt");
		double[] dropOffFor1stGain = singlePassAlgorithm(percentReduce, net, training, "g1");
		double[] dropOffFor2ndGain = singlePassAlgorithm(percentReduce, net, training, "g2");
		
		double[] algoForGT = bruteFuckingForce(percentReduce, net, training, "gt");
		double[] algoFor1G = continuousReestimationAlgorithm(percentReduce, net, training, "g1");
		double[] algoFor2G = continuousReestimationAlgorithm(percentReduce, net, training, "g2");

		int[][] combined = getRankingsMatrix(groundTruthRankings, gainSumRankings, secondGainSumRankings);
		double[][] combinedError = getErrorRankingsMatrix(groundTruthErrorRank, gainSumErrorRank, secondGainSumErrorRank);
		double[][] combinedDropoff = getErrorRankingsMatrix(dropOffForGT, dropOffFor1stGain, dropOffFor2ndGain);
		double[][] algoCombined = getErrorRankingsMatrix(algoForGT,algoFor1G,algoFor2G);

		String result = printMatrix(combined);
		String errResult = printErrorMatrix(combinedError);
		String dropOffResult = printErrorMatrix(combinedDropoff);
		String algoResult = printErrorMatrix(algoCombined);

//		System.out.println(result);
//		System.out.println(errResult);
		System.out.println(dropOffResult);
		System.out.println(algoResult);

		FileWriter writer;
		writer = new FileWriter(new File(dnnFile + ".ranking-result.csv"));
		writer.write(result);
		writer.close();
		
		writer = new FileWriter(new File(dnnFile + ".e0-e1-e2-change-result.csv"));
		writer.write(errResult);
		writer.close();
		
		writer = new FileWriter(new File(dnnFile + ".accuracy-dropoff-comparison.csv"));
		writer.write(dropOffResult);
		writer.close();

		writer = new FileWriter(new File(dnnFile + ".greedy-algo-comparison.csv"));
		writer.write(algoResult);
		writer.close();
		return net;
	}

	private static double[] bruteFuckingForce(double percentReduce, NeuralNetwork net, List<DataInstance> trainingSet, String sortBy) {
		List<GainSwitchNeuron> neurons = getGainSwitchNeurons(net);
		int neuronsToRemove = (int) Math.floor(neurons.size() * percentReduce);
		DNNTrainingModule trainingModule = initTrainingModule(net, trainingSet, trainingSet);
		double[] result = new double[neuronsToRemove];
		// turn everything on
		switchOffNeurons(neurons, false);
		for (int i = 0; i < neuronsToRemove; i++) {
			// Run through training set O(n * k)
			print("Running through training data 2nd derivative backprop...");
			getGroundTruthError(neurons, net, trainingSet);
			// Get the item at the top O(n)
			print("Picking the winner...");
			GainSwitchNeuron neuron = getBest(sortBy, neurons);
			// Switch it off
			System.out.println("Switching neuron " + neuron.getIdNum() + " OFF...");
			neuron.setSwitchOff(true);
			neurons.remove(neuron);
			// test network and get error
			double newError = trainingModule.doTestTrainedNetwork();
			result[i] = newError;
		}
		// switch back on
		for (GainSwitchNeuron neuron : neurons) {
			neuron.setSwitchOff(false);
		}
		return result;
	}

	private static double[] continuousReestimationAlgorithm(double percentReduce, NeuralNetwork net, List<DataInstance> trainingSet, String sortBy) {
		List<GainSwitchNeuron> sortedNeurons = getGainSwitchNeurons(net);
		int neuronsToRemove = (int) Math.floor(sortedNeurons.size() * percentReduce);
		DNNTrainingModule trainingModule = initTrainingModule(net, trainingSet, trainingSet);
		double initialError = trainingModule.doTestTrainedNetwork();
		double[] result = new double[neuronsToRemove];
		// turn everything on
		switchOffNeurons(sortedNeurons, false);
		for (int i = 0; i < neuronsToRemove; i++) {
			// Reset sums and stuff
			print("Resetting sums...");
			reset(net);
			// Run through training set O(n * k)
			print("Running through training data 2nd derivative backprop...");
			trainingModule.doTrainNetworkUntilConvergence();
			// Get the item at the top O(n)
			print("Picking the winner...");
			GainSwitchNeuron neuron = getBest(sortBy, sortedNeurons);
			// Switch it off
			System.out.println("Switching neuron " + neuron.getIdNum() + " OFF...");
			neuron.setSwitchOff(true);
			sortedNeurons.remove(neuron);
			// test network and get error
			double newError = trainingModule.doTestTrainedNetwork();
			double diff = newError - initialError;
			System.out.println("E(o1): " + initialError + " E(0): " + newError + "\nE(0) - E(o1): " + diff);
			result[i] = newError;
		}
		// switch back on
		switchOffNeurons(getGainSwitchNeurons(net), false);
		return result;
	}

	private static GainSwitchNeuron getBest(String sortBy, List<GainSwitchNeuron> sortedNeurons) {
		GainSwitchNeuron neuron = null;
		if (sortBy.equals("gt")) {
			double val = Double.POSITIVE_INFINITY;
			for (GainSwitchNeuron n : sortedNeurons) {
				if (!n.isSwitchedOff()) {
					if (n.getGroundTruthError() < val) {
						val = n.getGroundTruthError();
						neuron = n;
					}
				} else {
					System.out.println("Neuron: " + n.getIdNum() + " is switched off!");
				}
			}
			return neuron;
		}
		/* 
		 * 
		 * 
		 * USING THRESHOLDING on the MAGNITUDE
		 * */
	 double[] vals = getSortByValues(sortBy, sortedNeurons);
	 for(int i = 0; i < vals.length; i++)
	   vals[i] = Math.abs(vals[i]);
   double threshold = Math.abs(mean());
   /**/
		if (sortBy.equals("g1")) {
			double val = Double.NEGATIVE_INFINITY;
			for (GainSwitchNeuron n : sortedNeurons) {
				if (!n.isSwitchedOff()) {
				  double gain = n.getTotalGain();
					if (gain > val && Math.abs(gain) <= threshold) {
						val = gain;
						neuron = n;
					}
				} else {
					System.out.println("Neuron: " + n.getIdNum() + " is switched off!");
				}
			}
		} else if (sortBy.equals("g2")) {
			double val = Double.NEGATIVE_INFINITY;
			for (GainSwitchNeuron n : sortedNeurons) {
				if (!n.isSwitchedOff()) {
				  double gain = n.getTotalSecondGain();
					if (gain > val && Math.abs(gain) <= threshold) {
						val = gain;
						neuron = n;
					}
				} else {
					System.out.println("Neuron: " + n.getIdNum() + " is switched off!");
				}
			}

		}
		return neuron;
	}
	
	private static double mean(double... vals) {
	  double mean = 0;
	  for(double d : vals)
	    mean += d;
	  return mean / vals.length;
	}
	
	private static double var(double mean, double... vals) {
	  double var = 0;
	  for(double d : vals)
	    var += Math.pow((d - mean), 2);
	  return var / vals.length;
	}

	private static int[] sortNeurons(String sortBy, List<GainSwitchNeuron> neurons) {
		int[] result = new int[neurons.size()];
		if (sortBy.equals("gt"))
			result = sortByGroundTruthError(neurons);
		else if (sortBy.equals("g1"))
			result = sortByGain(neurons);
		else if (sortBy.equals("g2"))
			result = sortBySecondGain(neurons);
		return result;
	}

	private static List<GainSwitchNeuron> getGainSwitchNeurons(NeuralNetwork net) {
		/* Get the GainSwitch neurons out */
		System.out.println("Getting array of GainSwitch neurons...");
		List<GainSwitchNeuron> neurons = new LinkedList<GainSwitchNeuron>();
		for (int i : net.getHiddenLayerIndices()) {
			Layer l = net.getLayer(i);
			for (NetworkElement e : l.getElements()) {
				if (e instanceof GainSwitchNeuron)
					neurons.add((GainSwitchNeuron) e);
			}
		}
		return neurons;
	}

	private static void reset(NeuralNetwork net) {
		for (int i : net.getHiddenLayerIndices()) {
			for (NetworkElement e : net.getLayer(i).getElements()) {
				if (e instanceof GainSwitchNeuron) {
					((GainSwitchNeuron) e).reset();
				}
			}
		}
		for (int i : net.getWeightMatrixIndices()) {
			for (NetworkElement e : net.getLayer(i).getElements()) {
				if (e instanceof SwitchEdge) {
					((SwitchEdge) e).reset();
				}
			}
		}
	}

	private static double[] singlePassAlgorithm(double percentReduce, NeuralNetwork net, List<DataInstance> testingSet, String sortBy) {
		/* If reducing by percentage, get the number of neurons to remove */
		DNNTrainingModule trainingModule = initTrainingModule(net, testingSet, testingSet);
		List<GainSwitchNeuron> neurons = getGainSwitchNeurons(net);
		sortNeurons(sortBy, neurons);
    switchOffNeurons(neurons, false);
		double initialError = trainingModule.doTestTrainedNetwork();
		int neuronsToRemove = (int) Math.floor(neurons.size() * percentReduce);
		double[] result = new double[neuronsToRemove];
		for (int i = 0; i < neuronsToRemove; i++) {
			GainSwitchNeuron neuron = neurons.get(i);
			System.out.println("Switching neuron " + neuron.getIdNum() + " OFF...");
			neuron.setSwitchOff(true);
			double newError = trainingModule.doTestTrainedNetwork();
			double diff = newError - initialError;
			System.out.println("E(o1): " + initialError + " E(0): " + newError + "\nE(0) - E(o1): " + diff);
			result[i] = newError;
		}
		// switch back on
		switchOffNeurons(neurons, false);
		return result;
	}

	private static double[] getSortByValues(String sortBy, List<GainSwitchNeuron> sortedNeurons) {
	  if(sortBy.equals("gt"))
	    return getGTSortByValues(sortedNeurons);
	  else if (sortBy.equals("g1"))
	    return get1GSortByValues(sortedNeurons);
	  else if (sortBy.equals("g2"))
	    return get2GSortByValues(sortedNeurons);
	  else return new double[sortedNeurons.size()];
	}
	
	private static double[] get2GSortByValues(List<GainSwitchNeuron> sortedNeurons) {
		double[] ranks = new double[sortedNeurons.size()];
		for (int i = 0; i < ranks.length; i++) {
			ranks[i] = sortedNeurons.get(i).getTotalSecondGain();
		}
		return ranks;
	}

	private static double[] get1GSortByValues(List<GainSwitchNeuron> sortedNeurons) {
		double[] ranks = new double[sortedNeurons.size()];
		for (int i = 0; i < ranks.length; i++) {
			ranks[i] = sortedNeurons.get(i).getTotalGain();
		}
		return ranks;
	}

	private static double[] getGTSortByValues(List<GainSwitchNeuron> sortedNeurons) {
		double[] ranks = new double[sortedNeurons.size()];
		for (int i = 0; i < ranks.length; i++) {
			ranks[i] = sortedNeurons.get(i).getGroundTruthError();
		}
		return ranks;
	}

	private static void getGroundTruthError(List<GainSwitchNeuron> sortedNeurons, NeuralNetwork net, List<DataInstance> testingSet) {
		DNNTrainingModule trainingModule = initTrainingModule(net, testingSet, testingSet);
		double initialError = trainingModule.doTestTrainedNetwork();
		System.out.println("Unmodified network error: " + initialError);
		for (GainSwitchNeuron neuron : sortedNeurons) {
			System.out.println("Switching neuron " + neuron.getIdNum() + " OFF...");
			neuron.setSwitchOff(true);
			double newError = trainingModule.doTestTrainedNetwork();
			System.out.println("Switching neuron " + neuron.getIdNum() + " back ON...");
			neuron.setSwitchOff(false);
			double diff = newError - initialError;
			System.out.println("E(o1): " + initialError + " E(0): " + newError + "\nE(0) - E(o1): " + diff);
			neuron.setGroundTruthError(newError);
		}
	}

	private static List<NetworkElement> doRandomPruning(List<GainSwitchNeuron> sortedNeurons, int neuronsToRemove) {
		List<Integer> randomIndices = new ArrayList<Integer>();
		for (int i = 0; i < sortedNeurons.size(); i++)
			randomIndices.add(i);
		Collections.shuffle(randomIndices);
		List<NetworkElement> switchOffs = new ArrayList<NetworkElement>();
		for (int i = 0; i < neuronsToRemove; i++) {
			switchOffs.add((NetworkElement) sortedNeurons.get(randomIndices.get(i)));
		}
		return switchOffs;
	}

	/* ====================== HELPER METHODS ====================== */
	/* ====================== HELPER METHODS ====================== */
	/* ====================== HELPER METHODS ====================== */

	public static void switchOffNeurons(List<GainSwitchNeuron> sortedNeurons, boolean b) {
		for (GainSwitchNeuron s : sortedNeurons)
			s.setSwitchOff(b);
	}

	private static DNNTrainingModule initTrainingModule(NeuralNetwork net, List<DataInstance> training, List<DataInstance> testing) {
		DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
		trainingModule.setOutputAdapter(new BinaryThresholdOutput());
		trainingModule.setConvergenceCriteria(-1, -1, true, 0, 1);
		return trainingModule;
	}

	/* Sort by the real ground truth error... */
	private static int[] sortByGroundTruthError(List<GainSwitchNeuron> neuronsToSort) {
		Collections.sort(neuronsToSort, new Comparator<GainSwitchNeuron>() {
			@Override
			public int compare(GainSwitchNeuron o1, GainSwitchNeuron o2) {
				if (o1.getGroundTruthError() > o2.getGroundTruthError()) {
					return 1;
				} else if (o1.getGroundTruthError() < o2.getGroundTruthError()) {
					return -1;
				} else {
					return 0;
				}
			}
		});
		return getRankedIds(neuronsToSort);
	}

	/* Sort by first derivative gain */
	private static int[] sortByGain(List<GainSwitchNeuron> neuronsToSort) {
		Collections.sort(neuronsToSort, new Comparator<GainSwitchNeuron>() {
			@Override
			public int compare(GainSwitchNeuron o1, GainSwitchNeuron o2) {
				if (o1.getTotalGain() > o2.getTotalGain()) {
					return -1;
				} else if (o1.getTotalGain() < o2.getTotalGain()) {
					return 1;
				} else {
					return 0;
				}
			}
		});
		return getRankedIds(neuronsToSort);
	}

	/* Sort by the second gain terms: E(0) - E(o1) */
	private static int[] sortBySecondGain(List<GainSwitchNeuron> neuronsToSort) {
		Collections.sort(neuronsToSort, new Comparator<GainSwitchNeuron>() {
			@Override
			public int compare(GainSwitchNeuron o1, GainSwitchNeuron o2) {
				if (o1.getTotalSecondGain() > o2.getTotalSecondGain()) {
					return -1;
				} else if (o1.getTotalSecondGain() < o2.getTotalSecondGain()) {
					return 1;
				} else {
					return 0;
				}
			}
		});
		return getRankedIds(neuronsToSort);
	}

	private static int[] getRankedIds(List<GainSwitchNeuron> sortedNeurons) {
		int[] rankings = new int[sortedNeurons.size()];
		for (int i = 0; i < rankings.length; i++)
			rankings[i] = sortedNeurons.get(i).getIdNum();
		return rankings;
	}

	private static int[][] getRankingsMatrix(int[]... rankings) {
		return rankings;
	}

	private static double[][] getErrorRankingsMatrix(double[]... rankings) {
		return rankings;
	}

	private static String printMatrix(int[][] matrix) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < matrix[0].length; i++) {
			String[] arr = new String[matrix.length];
			for (int j = 0; j < matrix.length; j++) {
				arr[j] = "" + matrix[j][i];
			}
			String s = String.join(",", arr);
			sb.append(s);
			sb.append("\n");
		}
		return sb.toString();
	}

	private static String printErrorMatrix(double[][] matrix) {
		StringBuffer sb = new StringBuffer();
		for (int i = 0; i < matrix[0].length; i++) {
			String[] arr = new String[matrix.length];
			for (int j = 0; j < matrix.length; j++) {
				arr[j] = "" + matrix[j][i];
			}
			String s = String.join(",", arr);
			sb.append(s);
			sb.append("\n");
		}
		return sb.toString();
	}

	private static NeuralNetwork modifyLayers(NeuralNetwork net) {
		int[] hiddenLayerIndices = net.getHiddenLayerIndices();
		int[] edgeLayerIndices = net.getWeightMatrixIndices();
		for (int i : edgeLayerIndices) {
			Layer oldLayer = net.getLayer(i);
			GainSwitchLayer newLayer = new GainSwitchLayer(oldLayer);
			newLayer.switchOff(true);
			net.modifyExistingLayer(oldLayer, newLayer);
		}
		Layer outputLayer = net.getLastLayer();
		GainSwitchLayer newOutputLayer = new GainSwitchLayer(outputLayer);
		net.modifyExistingLayer(outputLayer, newOutputLayer);
		for (int i : hiddenLayerIndices) {
			Layer oldLayer = net.getLayer(i);
			GainSwitchLayer newLayer = new GainSwitchLayer(oldLayer);
			newLayer.resetLayer();
			net.modifyExistingLayer(oldLayer, newLayer);
		}
		return net;
	}

	private static void print(String s) {
		System.out.println(s);
	}

	// public static void removeElementsAndPruneLayers(NeuralNetwork net,
	// List<NetworkElement>
	// elementsToRemove) {
	// net.removeElements(elementsToRemove);
	// }

}
