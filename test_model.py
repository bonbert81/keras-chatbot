import unittest
import probar_modelo


class MyTestCase(unittest.TestCase):
    def test_something(self):
        for indice in range(100):
            pregunta, respuesta, prediccion = probar_modelo.buscar_pregunta(indice)
            test_name = 'test_%s' % pregunta
            test = test_generator(respuesta, prediccion)
            setattr(MyTestCase, test_name, test)
        unittest.main()


def test_generator(a, b):
    def test(self):
        self.assertEqual(a, b)

    return test


if __name__ == '__main__':
    for indice in range(100):
        pregunta, respuesta, prediccion = probar_modelo.buscar_pregunta(indice)
        test_name = 'test_%s' % pregunta
        test = test_generator(respuesta, prediccion)
        setattr(MyTestCase, test_name, test)
    unittest.main()
