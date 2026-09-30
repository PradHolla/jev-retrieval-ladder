from ladder import data

data.download()
corpus = data.load_corpus()
queries = data.load_queries()
test = data.load_qrels("test")
dev = data.dev_qrels()

print(f"corpus docs:   {len(corpus)}")
print(f"test queries:  {len(test)}  ({sum(map(len, test.values())) / len(test):.2f} relevant/query)")
print(f"dev queries:   {len(dev)}  ({sum(map(len, dev.values())) / len(dev):.2f} relevant/query)")
print(f"relevance levels in test: {sorted({r for rels in test.values() for r in rels.values()})}")
print(f"all qrel queries have text: {set(test) | set(dev) <= set(queries)}")
