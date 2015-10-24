package mlsp.cs.cmu.edu.dnn.training;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;

import mlsp.cs.cmu.edu.dnn.elements.GainSwitchNeuron;
import mlsp.cs.cmu.edu.dnn.elements.NetworkElement;
import mlsp.cs.cmu.edu.dnn.structure.GainSwitchLayer;
import mlsp.cs.cmu.edu.dnn.structure.Layer;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;

public class GainSwitchPruningTool {

  public static NeuralNetwork doPruning(NeuralNetwork net, List<DataInstance> validationSet, double percentReduce) {
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
    Collections.sort(sortedNeurons,new Comparator<GainSwitchNeuron>() {
      @Override
      public int compare(GainSwitchNeuron o1, GainSwitchNeuron o2) {
        if(o1.getAverageGain() > o2.getAverageGain()) {
          return -1;
        } else if(o1.getAverageGain() < o2.getAverageGain()) {
          return 1;
        } else {
          return 0;
        }
      }
    });
    int neuronsToRemove = (int) Math.floor(sortedNeurons.size() * percentReduce);
    for(GainSwitchNeuron n : sortedNeurons.subList(sortedNeurons.size()-neuronsToRemove, sortedNeurons.size())) {
      n.setSwitchOff(true);
    }
    return net;
  }
  
  private static NeuralNetwork modifyLayers(NeuralNetwork net) {
    int[] hiddenLayerIndices = net.getHiddenLayerIndices();
    int[] edgeLayerIndices = net.getWeightMatrixIndices();
    for(int i : hiddenLayerIndices) {
      Layer oldLayer = net.getLayer(i);
      GainSwitchLayer newLayer = new GainSwitchLayer(oldLayer);
      newLayer.resetLayer();
      net.modifyExistingLayer(oldLayer, newLayer);
    }
    for(int i : edgeLayerIndices) {
      Layer oldLayer = net.getLayer(i);
      GainSwitchLayer newLayer = new GainSwitchLayer(oldLayer);
      newLayer.switchOff(true);
      net.modifyExistingLayer(oldLayer, newLayer);
    }
    return net;
  }

}
