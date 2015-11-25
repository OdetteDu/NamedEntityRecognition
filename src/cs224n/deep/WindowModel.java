package cs224n.deep;

import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;
import java.io.*;

public class WindowModel {

	private HashMap<String, String> baselineWordMap;
	protected SimpleMatrix L, W, U;
	public int windowSize, wordSize, hiddenSize;
	

	public WindowModel(int _windowSize, int _hiddenSize, double _lr) {
		baselineWordMap = new HashMap<String, String>();
	}

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		// TODO Add one more column for b
		int H = hiddenSize + 1; // fanOut = H
		int Cn = windowSize * FeatureFactory.allVecs.numRows(); // fanIn = nC
		int K = Datum.POSSIBLE_LABELS.length;
		double range = Math.sqrt(6) / Math.sqrt(H + Cn);
		W = SimpleMatrix.random(H, Cn, -range, range, new Random());
		U = SimpleMatrix.random(K, H, -range, range, new Random());
	}

	/**
	 * Simplest SGD training
	 */
	public void train(List<Datum> trainData) {
		this.baselineTrain(trainData);
		this.nnTrain(trainData);
	}

	public void test(List<Datum> testData) {
		this.baselineTest(testData);
		this.nnTest(testData);
	}

	private void baselineTrain(List<Datum> trainData) {
		for (Datum datum : trainData) {
			this.baselineWordMap.put(datum.word, datum.label);
		}
	}

	private void baselineTest(List<Datum> testData) {

		List<Prediction> output = new ArrayList<Prediction>();
		for (Datum datum : testData) {
			String word = datum.word;
			String predictedLabel;
			if (this.baselineWordMap.containsKey(word)) {
				predictedLabel = this.baselineWordMap.get(word);
			} else {
				predictedLabel = Datum.DEFAULT_LABEL;
			}
			output.add(new Prediction(word, datum.label, predictedLabel));
		}
		this.outputToFile("baseline.out", output);
	}
	
	private void nnTrain(List<Datum> trainData)
	{
		for (int i=2; i<trainData.size(); i++)
		{
			SimpleMatrix sm = getWordVector(trainData.get(i-2), trainData.get(i-1), trainData.get(i));
			sm.print();
		}
	}
	
	private SimpleMatrix getWordVector(Datum first, Datum second, Datum third)
	{
		if (second.word.equals(Datum.SEPARATE_WORD))
		{
			return null;
		}
		
		double[] firstVector;
		if (first.word.equals(Datum.SEPARATE_WORD))
		{
			firstVector = this.getWordVector(Datum.START_WORD);
		}
		else
		{
			firstVector = this.getWordVector(first.word);
		}
		
		double[] thirdVector;
		if (third.word.equals(Datum.SEPARATE_WORD))
		{
			thirdVector = this.getWordVector(Datum.END_WORD);
		}
		else
		{
			thirdVector = this.getWordVector(third.word);
		}
		
		double[] secondVector = this.getWordVector(second.word);
		
		// Concatenate the three vectors
		double[] finalVector = new double[firstVector.length + secondVector.length + thirdVector.length];
		for (int i=0; i<firstVector.length; i++)
		{
			finalVector[i] = firstVector[i];
		}
		
		for (int i=0; i<secondVector.length; i++)
		{
			finalVector[i+firstVector.length] = secondVector[i];
		}
		
		for (int i=0; i<thirdVector.length; i++)
		{
			finalVector[i+firstVector.length+secondVector.length] = thirdVector[i];
		}
		
		double [][] matrixData = {finalVector};
		SimpleMatrix sm = new SimpleMatrix(matrixData);
		return sm.transpose();
	}
	
	private double[] getWordVector(String word)
	{
		int index;
		if(FeatureFactory.wordToNum.containsKey(word))
		{
			index = FeatureFactory.wordToNum.get(word);
		}
		else
		{
			index = FeatureFactory.NON_EXISTING_VOCAB_INDEX;
		}
		double[] vector = new double[FeatureFactory.allVecs.numCols()];
		for (int i=0; i<vector.length; i++)
		{
			vector[i] = FeatureFactory.allVecs.get(index, i);
		}
		return vector;
	}
	
	private void nnTest(List<Datum> testData)
	{
		
	}

	public void outputToFile(String fileName, List<Prediction> predictions) {

		String outputContent = "";
		for (Prediction prediction : predictions)
		{
			outputContent += prediction + "\n";
		}

		try 
		{
			File file = new File(fileName);
			if (!file.exists()) 
			{
				file.createNewFile();
			}
			BufferedWriter bw = new BufferedWriter(new FileWriter(file));
			bw.write(outputContent);
			bw.close();
		} 
		catch (IOException e) 
		{
			e.printStackTrace();
		}
	}

}
