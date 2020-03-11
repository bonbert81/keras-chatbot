import unittest
import probar_modelo


class MyTestCase(unittest.TestCase):

    def test_works_as_expected(self):
        for indice in range(20):
            pregunta, respuesta, prediccion = probar_modelo.buscar_pregunta(indice)
            respuesta = respuesta.replace('\t', '')
            respuesta = respuesta.replace('\n', '')
            test_name = 'test_%s' % pregunta
            with self.subTest():
                self.assertEqual(msg=test_name, first=respuesta, second=prediccion)

    # setattr(MyTestCase, "test_%r" % (pregunta), ch(respuesta, prediccion))
