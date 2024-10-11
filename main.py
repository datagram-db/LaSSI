from LaSSI.LaSSI import LaSSI

if __name__ == '__main__':
    pipeline = LaSSI("test_sentences/multiple_sentences.yaml", "connection.yaml")
    pipeline.run()
    pipeline.close()
