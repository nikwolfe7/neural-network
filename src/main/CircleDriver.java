package main;

import java.util.List;

import training.DNNTrainingModule;
import training.DataInstance;
import training.DataReader;
import training.ReadCSVTrainingData;
import mlsp.cs.cmu.edu.factory.DNNFactory;
import mlsp.cs.cmu.edu.factory.LinearOutputFeedForwardDNNFactory;
import mlsp.cs.cmu.edu.factory.SimpleFeedForwardDNNFactory;
import mlsp.cs.cmu.edu.factory.ThresholdOutputFFDNNFactory;
import mlsp.cs.cmu.edu.structure.NeuralNetwork;

public class CircleDriver {
  
  public static void main(String[] args) {
    
    DataReader reader = new ReadCSVTrainingData();
    List<DataInstance> training = reader.getDataFromFile("circle-train.csv", 2, 1);
    List<DataInstance> testing = reader.getDataFromFile("circle-test.csv", 2, 1);
    DNNFactory factory = new LinearOutputFeedForwardDNNFactory(training.get(0), 100);
    
    NeuralNetwork net = factory.getInitializedNeuralNetwork();
    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
    trainingModule.setOutputOn(true);
    trainingModule.doTrainNetworkUntilConvergence();
    trainingModule.doTestTrainedNetwork();
    
  }

}
