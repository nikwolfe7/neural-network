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
    DNNTrainingModule trainingModule = initTrainingModule(net, validationSet);
    trainingModule.doTrainNetworkUntilConvergence();
    List<GainSwitchNeuron> neuronsToSort = new ArrayList<GainSwitchNeuron>();
    for(int i : net.getHiddenLayerIndices()) {
      Layer l = net.getLayer(i);
      for(NetworkElement e : l.getElements()) {
        neuronsToSort.add((GainSwitchNeuron) e);
      }
    }
    int neuronsToRemove = (int) Math.floor(neuronsToSort.size() * percentReduce);
    sortByGain(neuronsToSort);
    List<GainSwitchNeuron> groundTruthSortedNeurons = getGroundTruthError(neuronsToSort, trainingModule);
    FileWriter writer = new FileWriter(new File("result.csv"));
    for(int i = 0; i < neuronsToSort.size(); i++) {
      String s = neuronsToSort.get(i).getIdNum() + "," + groundTruthSortedNeurons.get(i).getIdNum();
      writer.write(s + "\n");
      System.out.println(s);
    }
    writer.close();
//    List<NetworkElement> switchOff = doRankedPruning(neuronsToSort, neuronsToRemove);
//    List<NetworkElement> switchOff = doRandomPruning(sortedNeurons, neuronsToRemove);
//    if (removeElements)
//    	removeElementsAndPruneLayers(net, switchOff);
//    else
//    	switchElements(switchOff, true);
    return net;
  }
  
  
  
  
  
  
  
  private static List<GainSwitchNeuron> getGroundTruthError(List<GainSwitchNeuron> neuronsToSort, DNNTrainingModule trainingModule) {
    double initialError = trainingModule.doTestTrainedNetwork();
    List<GainSwitchNeuron> copyList = new ArrayList<GainSwitchNeuron>();
    for(GainSwitchNeuron neuron : neuronsToSort) {
      neuron.setSwitchOff(true);
      double newError = trainingModule.doTestTrainedNetwork();
      neuron.setSwitchOff(false);
      double diff = newError - initialError;
      neuron.setGroundTruthError(diff);
      copyList.add(neuron);
    }
    sortByGroundTruthError(copyList);
    return copyList;
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
  
  private static void sortByGroundTruthError(List<GainSwitchNeuron> neuronsToSort) {
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
  }
  
  private static void sortByGain(List<GainSwitchNeuron> neuronsToSort) {
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
  }
  
  private static void sortBySecondGain(List<GainSwitchNeuron> neuronsToSort) {
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
