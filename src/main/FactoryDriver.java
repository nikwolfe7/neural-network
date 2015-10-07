package main;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import mlsp.cs.cmu.edu.dnn.factory.DNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.FeedForwardDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.LinearOutputFeedForwardDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.SigmoidNetworkAbstractFactoryImpl;
import mlsp.cs.cmu.edu.dnn.factory.SimpleFeedForwardDNNFactory;
import mlsp.cs.cmu.edu.dnn.factory.ThresholdOutputFFDNNFactory;
import mlsp.cs.cmu.edu.dnn.structure.NeuralNetwork;
import mlsp.cs.cmu.edu.dnn.training.AdditionDataGenerator;
import mlsp.cs.cmu.edu.dnn.training.CosineGenerator;
import mlsp.cs.cmu.edu.dnn.training.DNNTrainingModule;
import mlsp.cs.cmu.edu.dnn.training.DataInstance;
import mlsp.cs.cmu.edu.dnn.training.DataInstanceGenerator;
import mlsp.cs.cmu.edu.dnn.training.MultipleOutputAddition;
import mlsp.cs.cmu.edu.dnn.training.XORGenerator;
import mlsp.cs.cmu.edu.dnn.util.DNNUtils;

public class FactoryDriver {

  static DataInstanceGenerator dataGen = new CosineGenerator();
  static DNNFactory factory = new SimpleFeedForwardDNNFactory(dataGen.getNewDataInstance(), 150);
  
  public static void main(String[] args) {
    
    /* Build the network */
    NeuralNetwork net = factory.getInitializedNeuralNetwork();

    /* Generate training and test data */
    List<DataInstance> training = getData(100000);
    List<DataInstance> testing = getData(10);

    /* Train the network */
    DNNTrainingModule trainingModule = new DNNTrainingModule(net, training, testing);
    trainingModule.setOutputOn(true);
    trainingModule.doTrainNetworkUntilConvergence();

    /* Test the network */
    trainingModule.doTestTrainedNetwork();

  }

  private static List<DataInstance> getData(int numInstances) {
    List<DataInstance> data = new ArrayList<>();
    for (int i = 0; i < numInstances; i++)
      data.add(dataGen.getNewDataInstance());
    return data;
  }
}
