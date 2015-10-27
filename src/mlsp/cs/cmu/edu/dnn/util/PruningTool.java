package mlsp.cs.cmu.edu.dnn.util;

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

  public static NeuralNetwork doPruning(NeuralNetwork net, List<DataInstance> validationSet, double percentReduce, boolean removeElements) {
    DNNTrainingModule trainingModule = new DNNTrainingModule(modifyLayers(net), validationSet, validationSet);
    trainingModule.setConvergenceCriteria(-1, -1, true, 0, 1);
    trainingModule.doTrainNetworkUntilConvergence();
    List<GainSwitchNeuron> sortedNeurons = new ArrayList<GainSwitchNeuron>();
    for(int i : net.getHiddenLayerIndices()) {
      Layer l = net.getLayer(i);
      for(NetworkElement e : l.getElements()) {
        sortedNeurons.add((GainSwitchNeuron) e);
      }
    }
    int neuronsToRemove = (int) Math.floor(sortedNeurons.size() * percentReduce);
    List<NetworkElement> switchOff = doRankedPruning(sortedNeurons, neuronsToRemove);
//    List<Switchable> switchOff = doRandomPruning(sortedNeurons, neuronsToRemove);
    switchElements(switchOff, true);
//    removeElementsAndPruneLayers(net, switchOff);
    return net;
  }
  
  public static void switchElements(List<NetworkElement> switchOff, boolean b) {
    for(NetworkElement s : switchOff) 
      ((Switchable) s).setSwitchOff(b);
  }
  
  public static void removeElementsAndPruneLayers(NeuralNetwork net, List<NetworkElement> elementsToRemove) {
    net.removeElements(elementsToRemove);
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
  
  private static void sortByGain(List<GainSwitchNeuron> neuronsToSort) {
    Collections.sort(neuronsToSort, new Comparator<GainSwitchNeuron>() {
      @Override
      public int compare(GainSwitchNeuron o1, GainSwitchNeuron o2) {
        if (o1.getAverageGain() > o2.getAverageGain()) {
          return -1;
        } else if (o1.getAverageGain() < o2.getAverageGain()) {
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
    for (int i : hiddenLayerIndices) {
      Layer oldLayer = net.getLayer(i);
      GainSwitchLayer newLayer = new GainSwitchLayer(oldLayer);
      newLayer.resetLayer();
      net.modifyExistingLayer(oldLayer, newLayer);
    }
    for (int i : edgeLayerIndices) {
      Layer oldLayer = net.getLayer(i);
      GainSwitchLayer newLayer = new GainSwitchLayer(oldLayer);
      newLayer.switchOff(true);
      net.modifyExistingLayer(oldLayer, newLayer);
    }
    return net;
  }

}
