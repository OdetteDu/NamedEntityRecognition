package cs224n.deep;

import java.util.*;

import org.ejml.data.*;
import org.ejml.simple.*;

import java.text.*;
import java.io.*;

public class WindowModel {

	private HashMap<String, String> baselineWordMap;
	protected SimpleMatrix L, W, U;
	public int windowSize;
	public int wordVectorSize;
	public int hiddenSize;


	public WindowModel(int _windowSize, int _hiddenSize, double _lr) {
		this.baselineWordMap = new HashMap<String, String>();
		this.windowSize = _windowSize;
		this.hiddenSize = _hiddenSize;
		this.wordVectorSize = FeatureFactory.allVecs.numRows();
	}

	/**
	 * Initializes the weights randomly.
	 */
	public void initWeights() {
		int H = hiddenSize; // fanOut = H
		int Cn = windowSize * wordVectorSize; // fanIn = nC
		int K = Datum.POSSIBLE_LABELS.length;
		double range = Math.sqrt(6) / Math.sqrt(H + Cn);
		W = SimpleMatrix.random(H, Cn + 1, -range, range, new Random());
		U = SimpleMatrix.random(K, H + 1, -range, range, new Random());
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
		System.out.println("W: "+W.numRows()+" * "+W.numCols());
		System.out.println("U: "+U.numRows()+" * "+U.numCols());
		for (int i=windowSize-1; i<trainData.size(); i++)
		{
			String[] wordsInWindow = new String[windowSize];
			int index = 0;
			for (int j=i-(windowSize-1); j<=i; j++)
			{
				wordsInWindow[index] = trainData.get(j).word;
				index++;
			}
			SimpleMatrix X = getWordVector(wordsInWindow);
			if (X != null)
			{
//				System.out.println("X: "+X.numRows()+" * "+X.numCols());
				SimpleMatrix WX = W.mult(X);
			}
		}
	}

	private SimpleMatrix getWordVector(String[] wordsInWindow)
	{
		if (wordsInWindow.length < 2)
		{
			return null;
		}
		
		for (int i=1; i<wordsInWindow.length-1; i++)
		{
			if (wordsInWindow[i].equals(Datum.SEPARATE_WORD))
			{
				return null;
			}
		}

		double[] firstVector;
		if (wordsInWindow[0].equals(Datum.SEPARATE_WORD))
		{
			firstVector = this.getWordVector(Datum.START_WORD);
		}
		else
		{
			firstVector = this.getWordVector(wordsInWindow[0]);
		}

		double[] lastVector;
		if (wordsInWindow[wordsInWindow.length-1].equals(Datum.SEPARATE_WORD))
		{
			lastVector = this.getWordVector(Datum.END_WORD);
		}
		else
		{
			lastVector = this.getWordVector(wordsInWindow[wordsInWindow.length-1]);
		}

		double[] finalVector = new double[wordVectorSize * wordsInWindow.length + 1];

		// Concatenate the three vectors
		for (int i=0; i<firstVector.length; i++)
		{
			finalVector[i] = firstVector[i];
		}

		for (int i=1; i<wordsInWindow.length-1; i++)
		{
			double[] middleVector = this.getWordVector(wordsInWindow[i]);
			for (int j=0; j<middleVector.length; j++)
			{
				finalVector[wordVectorSize * i + j] = middleVector[j];
			}
		}

		for (int i=0; i<lastVector.length; i++)
		{
			finalVector[i+wordVectorSize * (wordsInWindow.length-1)] = lastVector[i];
		}

		//Bias term
		finalVector[finalVector.length-1] = 1;

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
		double[] vector = new double[wordVectorSize];
		for (int i=0; i<vector.length; i++)
		{
			vector[i] = FeatureFactory.allVecs.get(i, index);
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
