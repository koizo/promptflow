#!/usr/bin/env python3
"""
Coverage test runner for DataCombiner executor.
"""

import coverage
import sys
import os

# Start coverage
cov = coverage.Coverage()
cov.start()

# Import and run our test
try:
    from test_data_combiner_minimal import main
    import asyncio
    
    # Run the tests
    result = asyncio.run(main())
    
    # Stop coverage and save
    cov.stop()
    cov.save()
    
    # Generate coverage report
    print("\n" + "="*60)
    print("COVERAGE ANALYSIS")
    print("="*60)
    
    # Show coverage for our DataCombiner implementation
    cov.report(show_missing=True, include="*data_combiner*")
    
    # Generate HTML report
    cov.html_report(directory='coverage_html', include="*data_combiner*")
    print(f"\nHTML coverage report generated in: coverage_html/")
    
except Exception as e:
    cov.stop()
    print(f"Error running coverage: {e}")
    sys.exit(1)
