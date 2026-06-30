from src.processor import DataProcessor
import logging

# Configuração mínima de log para ver o que está acontecendo
logging.basicConfig(level=logging.INFO)

def testar_pipeline_completo():
    print("\n--- TESTE DE INTEGRAÇÃO: Orquestrador ---")
    try:
        # Inicializa o orquestrador
        processor = DataProcessor()
        
        # BBOX de teste (área no Matopiba)
        bbox = [-45.21, -8.45, -45.15, -8.39]
        
        # Executa o pipeline (passamos uma lista vazia de geometrias para o teste inicial)
        # O processador deve baixar o tensor e processar os tiles
        tiles = processor.run_full_pipeline(bbox, "2026-05-01/2026-06-01", [])
        
        print(f"\n[SUCESSO] Pipeline finalizado!")
        print(f"Total de tiles gerados: {len(tiles)}")
        if len(tiles) > 0:
            print(f"Dimensão de um tile de cena: {tiles[0][0].shape}")
            print(f"Dimensão de um tile de máscara: {tiles[0][1].shape}")
        
    except Exception as e:
        print(f"\n[ERRO] O pipeline falhou no teste de integração: {e}")

if __name__ == "__main__":
    testar_pipeline_completo()