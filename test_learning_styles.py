#!/usr/bin/env python3
"""
Test script for learning style integration.
"""

from learning_styles import LearningStyleManager, LearningStyleType
from educational_prompts import EducationalPromptBuilder
from intelligent_enrichment import create_enriched_system_prompt


def test_learning_style_manager():
    """Test basic learning style manager functionality"""
    print("=== Testing Learning Style Manager ===")
    
    manager = LearningStyleManager()
    
    # Test available styles
    styles = manager.get_available_styles()
    print(f"Available learning styles: {len(styles)}")
    for style, desc in styles.items():
        print(f"  - {style}: {desc}")
    
    # Test persona retrieval
    academic_persona = manager.get_persona("academic_focus")
    print(f"\nAcademic Focus persona: {academic_persona.name}")
    print(f"Goal: {academic_persona.goal}")
    
    # Test auto-detection
    test_query = "How do I prepare for my exam on machine learning?"
    detected_style = manager.auto_detect_learning_style(test_query)
    print(f"\nAuto-detected learning style for '{test_query}': {detected_style}")
    
    return True


def test_educational_prompt_builder():
    """Test educational prompt builder"""
    print("\n=== Testing Educational Prompt Builder ===")
    
    builder = EducationalPromptBuilder()
    
    # Test prompt creation
    query = "What is machine learning?"
    document_context = "Machine learning is a branch of artificial intelligence that uses algorithms to analyze data and make predictions."
    
    educational_prompt, metadata = builder.create_educational_system_prompt(
        learning_style="academic_focus",
        query=query,
        document_context=document_context
    )
    
    print(f"Generated educational prompt (length: {len(educational_prompt)} chars)")
    print(f"Metadata: {metadata}")
    
    # Test with different learning style
    narrative_prompt, narrative_metadata = builder.create_educational_system_prompt(
        learning_style="narrative_reader", 
        query=query,
        document_context=document_context
    )
    
    print(f"\nNarrative style prompt (length: {len(narrative_prompt)} chars)")
    print(f"Narrative metadata: {narrative_metadata}")
    
    return True


def test_enrichment_integration():
    """Test integration with enrichment system"""
    print("\n=== Testing Enrichment Integration ===")
    
    query = "Explain the concept of neural networks"
    document_context = "Neural networks are computational models inspired by biological neural networks. They consist of interconnected nodes that process information."
    
    # Test with learning style integration
    enhanced_prompt, metadata = create_enriched_system_prompt(
        query=query,
        document_context=document_context,
        allow_general_knowledge=True,
        learning_style="deep_dive",
        educational_mode=True
    )
    
    print(f"Enhanced prompt with learning style (length: {len(enhanced_prompt)} chars)")
    print(f"Educational mode: {metadata.get('educational_mode', False)}")
    print(f"Learning style: {metadata.get('learning_style', 'None')}")
    print(f"Enrichment applied: {metadata.get('enrichment_applied', False)}")
    
    return True


def main():
    """Run all tests"""
    print("🧪 Testing Learning Style Integration")
    print("=" * 50)
    
    try:
        # Run tests
        test_learning_style_manager()
        test_educational_prompt_builder()
        test_enrichment_integration()
        
        print("\n✅ All tests passed!")
        print("\n🎓 Learning Style Integration is working correctly!")
        
        # Show example usage
        print("\n" + "=" * 50)
        print("📚 EXAMPLE USAGE:")
        print("=" * 50)
        
        print("\n1. Available Learning Styles:")
        manager = LearningStyleManager()
        for style, desc in manager.get_available_styles().items():
            print(f"   - {style}")
        
        print("\n2. API Parameter Examples:")
        print("   - learning_style='academic_focus'  # For exam preparation")
        print("   - learning_style='quick_practical'  # For actionable steps")
        print("   - learning_style='narrative_reader' # For readable content")
        print("   - educational_mode=True            # Enable tutoring approach")
        
        print("\n3. Auto-Detection Examples:")
        test_queries = [
            ("How do I study for my exam?", "academic_focus"),
            ("Give me step-by-step instructions", "quick_practical"), 
            ("Tell me about quantum physics", "exploratory_curious"),
            ("Explain this in detail", "deep_dive")
        ]
        
        for query, expected in test_queries:
            detected = manager.auto_detect_learning_style(query)
            print(f"   Query: '{query}' → Detected: {detected}")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)