import unittest
import BayesianDataStructure
import jgraph

class mytest(unittest.TestCase):
    def testInit(self):
        node1 = BayesianDataStructure.Node("CarType", 10000, ["suv", "sedan"], [7000, 3000])

    def testDataParser(self):
        dp = BayesianDataStructure.DataParser()
        dp.readCsvFile(r"D:\projects\data_compression\BayesianNetworkPython\DATA\AggUsage_OneDriveForBusiness_Aggregated_selectColumns.txt", True, 10000)
        Measurements = map(lambda x: float(x), dp.ColumnVals[-1])
        bynt = BayesianDataStructure.BayesianNetwork(dp.ColumnNames[:-1], dp.ColumnVals[:-1], Measurements)
        
        bynt.Draft()
        print bynt.BasicGraph
        bynt.Thickening()
        print bynt.BasicGraph

    def oldtestCMI(self):
        vals = [["chn", "chn", "us"], ["suv", "suv", "crv"], ["small", "big", "small"]]
        #col0 col1 0.5004024, col0 col2 == col1 col2
        bynt = BayesianDataStructure.BayesianNetwork(["col0", "col1", "col2"], vals, [300, 500, 200])
        bynt.Draft()
        #print bynt.BasicGraph
        bynt.Thickening()
        #print bynt.BasicGraph
        print bynt.BasicGraph.Connections
        self.assertEqual("%.7f" % bynt.BasicGraph.Connections[0][1], "0.7219281")
        self.assertEqual(bynt.BasicGraph.Connections[0][2], bynt.BasicGraph.Connections[1][2])

    def testCMINew(self):
        vals = [["chn", "us", "chn", "us"], ["suv", "suv", "crv", "crv"], [300, 300, 300, 100]]
        bynt = BayesianDataStructure.BayesianNetwork(["col0", "col1"], vals[:2], vals[-1])
        print bynt.ConditionalMutualInformation(bynt.Nodes[0], bynt.Nodes[1], [])
        bynt.Draft()
        bynt.Thickening()
        bynt.Thinning()

    def testDrawGraph(self):
        #jgraph.draw([(1,2), (2,3), (3,4)])
        pass

if __name__ == "__main__":
    unittest.main()
