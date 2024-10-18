import PySimpleGUI as sg
from mne.decoding import Scaler 

class Scaler_modal:

    """
        Janela modal para a configuração de um escalonador
         priors=None
         covariance_estimator=None
    """

    def __init__(self):
        self._names_column = [  [sg.Text('Escala')],
                                [sg.Text('Com média')],
                                [sg.Text('Com desvio padrão')],
                                ]
        self._entries_column = [[sg.Combo(['-', 'mean', 'median'], default_value='-', key='-scaler_method-', size=(20,1))],
                                [sg.Combo(['True','False'], default_value='True', key='-scaler_with_mean-', size=(20,1))],
                                [sg.Combo(['True','False'], default_value='True', key='-scaler_with_std-', size=(20,1))],
                                ]
        self._layout = [[sg.Column(self._names_column), sg.Column(self._entries_column)],
                        [sg.Text('',key='-warning-',visible=False)],
                        [sg.Button("CANCELAR", key='-cancel-', button_color="#ff0000"), sg.Button("PRONTO", key='-ok-')]]

    def open_window(self):
        window = sg.Window("Parâmetros escalonador", self._layout, modal=True)

        while True:
            event, values = window.read()
            
            if event == sg.WIN_CLOSED or event == '-cancel-':
                window.close()
                return ('cancel')
            if event == '-ok-':
                validation = self._validate_values(values)
                if validation != '':
                    window['-warning-'].Update(validation, visible=True)
                else:
                    window.close()
                    return self._build_method(values)

    def _validate_values(self, values):

        """
        Validação dos valores dentro da modal, garante que 
        os dados inseridos seguem o padrão necessário da
        aplicação. 
        """

        if values['-scaler_method-'] not in ['-', 'mean', 'median']:
            return 'Escalonador inválido.'
            
        if values['-scaler_with_mean-'] not in ['True', 'False']:
            return 'Valor de média deve ser True ou False'

        if values['-scaler_with_std-'] not in ['True', 'False']:
            return 'Valor de desvio padrão deve ser True ou False'

        return ''

    def _build_method(self, values):
        
        if values['-scaler_method-'] == '-':
            values['-scaler_method-'] = None

        if values['-scaler_with_mean-'] == 'True':
            values['-scaler_with_mean-'] = True
        else:
            values['-scaler_with_mean-'] = False

        if values['-scaler_with_std-'] == 'True':
            values['-scaler_with_std-'] = True
        else:
            values['-scaler_with_std-'] = False

        sca = Scaler(scalings=values['-scaler_method-'],
                     with_mean=values['-scaler_with_mean-'],
                     with_std=values['-scaler_with_std-'])

        return {'method': ("Scaler", sca), 'args': values}
        
        