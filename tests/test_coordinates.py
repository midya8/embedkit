"""Finite coordinate and event-order consistency checks, not a correctness oracle."""
import copy
from importlib import resources
import json
import random
import unittest

from embedkit.inspector import inspect
from embedkit.model import Embeddings


def close(test, a, b):
    if isinstance(a, (list,tuple)):
        test.assertEqual(len(a),len(b))
        for x,y in zip(a,b):close(test,x,y)
    else:test.assertLessEqual(abs(a-b),2e-11*max(1,abs(a),abs(b)))


def transformed(value, permutation, signs):
    result=copy.deepcopy(value)
    result['center']=[signs[i]*value['center'][j] for i,j in enumerate(permutation)]
    result['outputs']=[[signs[i]*row[j] for i,j in enumerate(permutation)] for row in value['outputs']]
    return result


def mapped(vector, permutation, signs):
    return [signs[i]*vector[j] for i,j in enumerate(permutation)]


class CoordinateTests(unittest.TestCase):
    def test_all_archived_reports_under_matching_signed_coordinates(self):
        data=resources.files('embedkit').joinpath('data')
        witnesses=[w for i in range(6) for w in json.loads(data.joinpath(f'witness-{i:02d}.json').read_text())]
        count=0
        for witness in witnesses:
            value=witness['value'];a=inspect(value);dim=len(value['center'])
            for permutation in (list(reversed(range(dim))),list(range(1,dim))+[0]):
                signs=[-1 if i%2==0 else 1 for i in range(dim)]
                b=inspect(transformed(value,permutation,signs))
                for key in ('loss_before','update_l2_difference'):close(self,a[key],b[key])
                close(self,mapped(a['center_gradient'],permutation,signs),b['center_gradient'])
                close(self,[mapped(row,permutation,signs) for row in a['output_gradients']],b['output_gradients'])
                for mode in ('simultaneous','sequential'):
                    close(self,mapped(a[mode]['center'],permutation,signs),b[mode]['center'])
                    close(self,[mapped(row,permutation,signs) for row in a[mode]['outputs']],b[mode]['outputs'])
                    close(self,a[mode]['loss_after'],b[mode]['loss_after'])
                for x,y in zip(a['sequential']['trajectory'],b['sequential']['trajectory']):
                    self.assertEqual(x['was_clipped'],y['was_clipped'])
                    for key in ('dot','clipped_dot','factor','true_factor','event_loss_before','event_loss_after'):close(self,x[key],y[key])
                    for key in ('center_before','center_after','output_before','output_after'):close(self,mapped(x[key],permutation,signs),y[key])
                count+=1
        self.assertEqual(count,38)

    def test_generated_signed_updates_and_nonmatched_control(self):
        rng=random.Random(520);count=0
        for case in range(48):
            dim=1+case%8;rows=1+case%5
            value={'center':[rng.randint(-16,16)/8 for _ in range(dim)],
                   'outputs':[[rng.randint(-16,16)/8 for _ in range(dim)] for _ in range(rows)],
                   'events':[{'row':rng.randrange(rows),'label':rng.randrange(2)} for _ in range(1+case%9)],
                   'lr':[.01,.2,.8][case%3]}
            permutation=list(range(dim));rng.shuffle(permutation);signs=[rng.choice([-1,1]) for _ in range(dim)]
            a=inspect(value);b=inspect(transformed(value,permutation,signs))
            close(self,a['loss_before'],b['loss_before'])
            for mode in ('sequential','simultaneous'):
                close(self,mapped(a[mode]['center'],permutation,signs),b[mode]['center'])
                close(self,[mapped(row,permutation,signs) for row in a[mode]['outputs']],b[mode]['outputs'])
            count+=1
        self.assertEqual(count,48)
        value={'center':[1.,2.],'outputs':[[3.,1.]],'events':[{'row':0,'label':1}],'lr':.1}
        changed=copy.deepcopy(value);changed['center'][0]*=-1
        self.assertNotAlmostEqual(inspect(value)['loss_before'],inspect(changed)['loss_before'])

    def test_frozen_event_order_invariance_is_not_sequential_invariance(self):
        value={'center':[1.,-.5],'outputs':[[.7,.2],[-.3,1.]],
               'events':[{'row':0,'label':1},{'row':1,'label':0},{'row':0,'label':0}], 'lr':.4}
        a=inspect(value);reverse=copy.deepcopy(value);reverse['events'].reverse();b=inspect(reverse)
        close(self,a['loss_before'],b['loss_before'])
        close(self,a['center_gradient'],b['center_gradient']);close(self,a['output_gradients'],b['output_gradients'])
        for key in ('center','outputs','loss_after'):close(self,a['simultaneous'][key],b['simultaneous'][key])
        self.assertGreater(max(abs(x-y) for x,y in zip(a['sequential']['center'],b['sequential']['center'])),1e-4)
        self.assertNotAlmostEqual(a['sequential']['loss_after'],b['sequential']['loss_after'])

    def test_full_trained_cosine_table_signed_coordinates_not_general_scaling(self):
        f=json.loads(resources.files('embedkit').joinpath('data/original.json').read_text())
        vectors=f['vectors'];a=Embeddings(vectors);dim=a.dim
        permutation=list(reversed(range(dim)));signs=[-1 if i%2 else 1 for i in range(dim)]
        b=Embeddings({word:mapped(row,permutation,signs) for word,row in vectors.items()})
        count=0
        for word in sorted(vectors):
            expected=a.nearest(word,len(vectors)-1);actual=b.nearest(word,len(vectors)-1)
            self.assertEqual([w for w,_ in expected],[w for w,_ in actual])
            close(self,[v for _,v in expected],[v for _,v in actual])
            for other in sorted(vectors):close(self,a.similarity(word,other),b.similarity(word,other));count+=1
        self.assertEqual(count,1024)
        original=Embeddings({'a':[1.,1.],'b':[1.,0.]})
        scaled=Embeddings({'a':[2.,1.],'b':[2.,0.]})
        self.assertNotAlmostEqual(original.similarity('a','b'),scaled.similarity('a','b'))


if __name__=='__main__':unittest.main()
