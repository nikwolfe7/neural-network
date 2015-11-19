package mlsp.cs.cmu.edu.dnn.util;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import mlsp.cs.cmu.edu.dnn.elements.GainSwitchNeuron;
import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;
import mlsp.cs.cmu.edu.dnn.elements.Switchable;
import mlsp.cs.cmu.edu.dnn.structure.GainSwitchLayer;
import mlsp.cs.cmu.edu.dnn.structure.Layer;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;

public class PruningTool {

  public static NeuralNetwork doPruning(NeuralNetwork net, List<DataInstance> validationSet, double percentReduce, boolean removeElements) throws IOException {
    /* Get training module and run through the validation set once */
    DNNTrainingModule trainingModule = initTrainingModule(net, validationSet);
    trainingModule.doTrainNetworkUntilConvergence();
    
    /* Get the GainSwitch neurons out */
    List<GainSwitchNeuron> neuronsToSort = new ArrayList<GainSwitchNeuron>();
    for(int i : net.getHiddenLayerIndices()) {
      Layer l = net.getLayer(i);
      for(NetworkElement e : l.getElements()) {
        if(e instanceof GainSwitchNeuron) 
          neuronsToSort.add((GainSwitchNeuron) e);
      }
    }
    
    /* Get the ground truth for the change in error resulting from turning the neurons on and off */
    List<GainSwitchNeuron> groundTruthSortedNeurons = getGroundTruthError(neuronsToSort, trainingModule);
    
    /* If reducing by percentage, get the number of neurons to remove */
    int neuronsToRemove = (int) Math.floor(neuronsToSort.size() * percentReduce);
    
    /* Sort the list by the different metrics */
    int[] groundTruthRankings = sortByGroundTruthError(neuronsToSort);
    int[] gainSumRankings = sortByGain(neuronsToSort);
    int[] secondGainSumRankings = sortBySecondGain(neuronsToSort);
    
    int[][] combined = getRankingsMatrix(groundTruthRankings, gainSumRankings, secondGainSumRankings);
    String result = printMatrix(combined);
    System.out.println(result);
    
    FileWriter writer = new FileWriter(new File("result.csv"));
    
    writer.close();
//    List<NetworkElement> switchOff = doRankedPruning(neuronsToSort, neuronsToRemove);
//    List<NetworkElement> switchOff = doRandomPruning(sortedNeurons, neuronsToRemove);
//    if (removeElements)
//    	removeElementsAndPruneLayers(net, switchOff);
//    else
//    	switchElements(switchOff, true);
    return net;
  }
  
  
  
  
  
  
  
  

  private static List<NetworkElement> doRandomPruning(List<GainSwitchNeuron> sortedNeurons, int neuronsToRemove) {
    List<Integer> randomIndices = new ArrayList<Integer>();
    for(int i = 0; i < sortedNeurons.size(); i++)
      randomIndices.add(i);
    Collections.shuffle(randomIndices);
    List<NetworkElement> switchOffs = new ArrayList<NetworkElement>();
    for(int i = 0; i < neuronsToRemove; i++) {
        switchOffs.add((NetworkElement) sortedNeurons.get(randomIndices.get(i)));
    }
    return switchOffs;
  }
  
  private static List<NetworkElement> doRankedPruning(List<GainSwitchNeuron> sortedNeurons, int neuronsToRemove) {
    sortByGain(sortedNeurons);
    List<NetworkElement> switchOffs = new ArrayList<NetworkElement>();
    List<GainSwitchNeuron> subList = sortedNeurons.subList(sortedNeurons.size() - neuronsToRemove, sortedNeurons.size());
    for(GainSwitchNeuron n : subList) {
      switchOffs.add(n);
    }
    return switchOffs;
  }
  
  
  /* ====================== HELPER METHODS ====================== */
  /* ====================== HELPER METHODS ====================== */
  /* ====================== HELPER METHODS ====================== */
  
  private static List<GainSwitchNeuron> getGroundTruthError(List<GainSwitchNeuron> neuronsToSort, DNNTrainingModule trainingModule) {
    double initialError = trainingModule.doTestTrainedNetwork();
    for(GainSwitchNeuron neuron : neuronsToSort) {
      neuron.setSwitchOff(true);
      double newError = trainingModule.doTestTrainedNetwork();
      neuron.setSwitchOff(false);
      double diff = newError - initialError;
      neuron.setGroundTruthError(diff);
    }
    return neuronsToSort;
  }
  
  public static void switchElements(List<NetworkElement> switchOff, boolean b) {
    for(NetworkElement s : switchOff) 
      ((Switchable) s).setSwitchOff(b);
  }
  
  private static DNNTrainingModule initTrainingModule(NeuralNetwork net, List<DataInstance> validationSet) {
    DNNTrainingModule trainingModule = new DNNTrainingModule(modifyLayers(net), validationSet, validationSet);
    trainingModule.setOutputAdapter(new BinaryThresholdOutput());
    trainingModule.setConvergenceCriteria(-1, -1, true, 0, 1);
    trainingModule.doTestTrainedNetwork();
    return trainingModule;
  }
  
  /* Sort by the real ground truth error...  */
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
  
  private static int[] getRankedIds(List<GainSwitchNeuron> neuronsToSort) {
    int[] rankings = new int[neuronsToSort.size()];
    for(int i = 0; i < rankings.length; i++) 
      rankings[i] = neuronsToSort.get(i).getIdNum();
    return rankings; 
  }
  
  private static int[][] getRankingsMatrix(int[]... rankings) {
    return rankings;
  }
  
  private static String printMatrix(int[][] matrix) {
    StringBuffer sb = new StringBuffer();
    for(int i = 0; i < matrix.length; i++) {
      String[] arr = new String[matrix[i].length];
      for(int j = 0; j < matrix[0].length; j++) {
        arr[j] = "" + matrix[i][j];
      }
      sb.append(String.join(",", arr));
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

  //public static void removeElementsAndPruneLayers(NeuralNetwork net, List<NetworkElement> elementsToRemove) {
  //net.removeElements(elementsToRemove);
  //}
  
  
}
